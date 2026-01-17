import logging
import json
import re
from dataclasses import dataclass, asdict
from typing import Optional, List, Union, Dict, Any

# Import base classes and utilities
from ..utils import (
    BaseRapidAPI,
    BaseResponse,
    APIError,
    AuthenticationError,
    RequestError,
    InvalidInputError,
    APIConfig,
    with_retry,
)

logger = logging.getLogger(__name__)


# ==================== Exception Aliases (Backward Compatibility) ====================

# Create aliases for backward compatibility
class SuggestError(APIError):
    """Error raised when suggestion request fails"""
    def __init__(self, reason: str):
        super().__init__(f"Suggestion failed: {reason}")
        self.reason = reason


class APIResponseError(RequestError):
    """Error raised when the API does not return a 200 response or provides an error message"""
    def __init__(self, message: str):
        super().__init__(f"API Error: {message}")
        self.message = message


class ProcessingError(RequestError):
    """Error raised when response processing fails"""
    def __init__(self, reason: str):
        super().__init__(f"Processing failed: {reason}")
        self.reason = reason


# ==================== Data Models ====================

@dataclass
class SuggestionResult:
    """Result of YouTube search suggestions"""
    query: str
    suggestions: List[str]
    raw_response: Optional[str] = None
    success: bool = True

    @property
    def count(self) -> int:
        """Number of suggestions returned"""
        return len(self.suggestions)

    @property
    def has_suggestions(self) -> bool:
        """Check if any suggestions were found"""
        return self.count > 0

    def to_dict(self, include_raw: bool = False) -> Dict[str, Any]:
        """
        Convert to dictionary.

        Args:
            include_raw: If True, includes raw_response

        Returns:
            Dictionary representation
        """
        data = {
            "query": self.query,
            "suggestions": self.suggestions,
            "success": self.success,
            "count": self.count
        }
        if include_raw:
            data["raw_response"] = self.raw_response
        return data

    def to_json(self, indent: Optional[int] = None, include_raw: bool = False) -> str:
        """
        Convert to JSON string.

        Args:
            indent: Number of spaces for indentation (None for compact JSON)
            include_raw: If True, includes raw_response

        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(include_raw=include_raw), indent=indent, ensure_ascii=False)


# ==================== API Client ====================

class YouTubeSuggestAPI(BaseRapidAPI):
    """
    API wrapper for YouTube search suggestions.

    Inherits from BaseRapidAPI for common functionality including:
    - Session management
    - Header creation
    - Response validation
    - Error handling

    Example:
        async with YouTubeSuggestAPI(api_key="key") as client:
            result = await client.get_suggestions("python programming")
            print(result.suggestions)

            # Use with config
            config = APIConfig(api_key="key", max_retries=5)
            async with YouTubeSuggestAPI(config=config) as client:
                result = await client.get_suggestions("query")
    """

    BASE_URL = "https://youtube-suggest-api.p.rapidapi.com"
    DEFAULT_HOST = "youtube-suggest-api.p.rapidapi.com"

    # __init__, __aenter__, __aexit__, _get_headers inherited from BaseRapidAPI

    # -------------------- Response Processing --------------------

    def _process_google_response(self, response_string: str) -> List[str]:
        """
        Process Google's callback-style response and extract suggestions.

        Args:
            response_string (str): Raw response from API

        Returns:
            List[str]: List of search suggestions

        Raises:
            ProcessingError: If processing fails
        """
        try:
            # Extract JSON array from the callback format
            match = re.search(r'\[.*\]', response_string)
            if not match:
                raise ProcessingError("Could not find JSON array in response")

            # Parse the JSON array
            data_list = json.loads(match.group(0))

            # Extract suggestions array (second element)
            if len(data_list) < 2:
                raise ProcessingError("Response format unexpected - missing suggestions array")

            suggestions_array = data_list[1]

            # Clean and extract suggestion text
            cleaned_results = [item[0] for item in suggestions_array if isinstance(item, list) and len(item) > 0]

            return cleaned_results

        except json.JSONDecodeError as e:
            raise ProcessingError(f"Failed to parse JSON: {str(e)}")
        except (IndexError, KeyError) as e:
            raise ProcessingError(f"Unexpected response structure: {str(e)}")

    # -------------------- Request Handler --------------------

    async def _request_get(self, endpoint: str, params: dict) -> str:
        """
        Send GET request to API.

        Args:
            endpoint (str): API endpoint
            params (dict): Query parameters

        Returns:
            str: Raw response text

        Raises:
            APIResponseError: If request fails
        """
        if not self._session:
            raise APIError("Session not initialized. Use async context manager.")

        url = f"{self.BASE_URL}/{endpoint}"
        headers = self._get_headers()
        logger.info(f"GET {url} with params: {params}")

        try:
            async with self._session.get(url, headers=headers, params=params) as response:
                if response.status == 401 or response.status == 403:
                    raise AuthenticationError(
                        "Authentication failed",
                        status_code=response.status,
                        endpoint=endpoint
                    )

                if response.status != 200:
                    error_text = await response.text()
                    raise RequestError(
                        f"Request failed",
                        status_code=response.status,
                        response_text=error_text,
                        endpoint=endpoint
                    )

                return await response.text()

        except (AuthenticationError, RequestError):
            raise
        except Exception as e:
            logger.error(f"Request error: {str(e)}")
            raise RequestError(f"Network error: {str(e)}", endpoint=endpoint, original_error=e)

    # -------------------- Public Methods --------------------

    async def search(self, query: str, process_response: bool = True) -> Union[SuggestionResult, str]:
        """
        Get YouTube search suggestions for a query.

        Args:
            query (str): Search query
            process_response (bool): If True, process and clean the response.
                                   If False, return raw response string.
                                   Default: True

        Returns:
            SuggestionResult: If process_response=True, returns structured result
            str: If process_response=False, returns raw response string

        Raises:
            APIResponseError: If the API request fails
            ProcessingError: If process_response=True and processing fails
        """
        params = {"search_query": query}
        raw_response = await self._request_get("search", params)

        # If user wants raw response
        if not process_response:
            logger.info(f"Returning raw response for query: {query}")
            return raw_response

        # Process the response
        try:
            suggestions = self._process_google_response(raw_response)
            logger.info(f"Found {len(suggestions)} suggestions for query: {query}")

            return SuggestionResult(
                query=query,
                suggestions=suggestions,
                raw_response=raw_response,
                success=True
            )

        except ProcessingError as e:
            logger.error(f"Failed to process response for query '{query}': {e}")
            raise

    async def batch_search(self, queries: List[str], process_response: bool = True) -> List[
        Union[SuggestionResult, str]]:
        """
        Get suggestions for multiple queries.

        Args:
            queries (List[str]): List of search queries
            process_response (bool): If True, process responses. Default: True

        Returns:
            List: List of results (SuggestionResult or str depending on process_response)
        """
        results = []

        for i, query in enumerate(queries):
            try:
                result = await self.search(query, process_response)
                results.append(result)

                if process_response and isinstance(result, SuggestionResult):
                    logger.info(f"Query {i + 1}/{len(queries)}: '{query}' - {result.count} suggestions")
                else:
                    logger.info(f"Query {i + 1}/{len(queries)}: '{query}' - raw response returned")

            except Exception as e:
                logger.error(f"Failed to process query {i + 1} ('{query}'): {e}")
                if process_response:
                    results.append(SuggestionResult(
                        query=query,
                        suggestions=[],
                        success=False
                    ))
                else:
                    results.append("")

        return results
