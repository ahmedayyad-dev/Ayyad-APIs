"""
YouTube Suggest API wrapper for getting YouTube search suggestions.

This module provides a simple async interface to interact with YouTube Suggest API
through RapidAPI, allowing users to get search suggestions for any query.

Author: Ahmed Ayyad
"""

import logging
import json
import re
from dataclasses import dataclass
from typing import Optional, List, Union, Dict, Any

# Import base classes and utilities
from ..utils import (
    BaseRapidAPI,
    BaseResponse,
    APIError,
    RequestError,
)

logger = logging.getLogger(__name__)


# ==================== Exception Aliases (Backward Compatibility) ====================

SuggestError = APIError
APIResponseError = RequestError
ProcessingError = RequestError


# ==================== Data Models ====================

@dataclass
class SuggestionResult(BaseResponse):
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

    # to_dict() and to_json() inherited from BaseResponse

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SuggestionResult":
        """
        Create SuggestionResult from dictionary.

        Args:
            data: Dictionary with query, suggestions, etc.

        Returns:
            SuggestionResult instance
        """
        return cls(
            query=data.get("query", ""),
            suggestions=data.get("suggestions", []),
            raw_response=data.get("raw_response"),
            success=data.get("success", True)
        )


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

    # ==================== Response Processing ====================

    def _process_google_response(self, response_string: str) -> List[str]:
        """
        Process Google's callback-style response and extract suggestions.

        Args:
            response_string: Raw response from API

        Returns:
            List of search suggestions

        Raises:
            ProcessingError: If processing fails
        """
        try:
            # Extract JSON array from the callback format
            match = re.search(r'\[.*\]', response_string)
            if not match:
                raise ProcessingError("Could not find JSON array in response")

            # Parse the JSON array
            data_list: list = json.loads(match.group(0))

            # Extract suggestions array (second element)
            if len(data_list) < 2:
                raise ProcessingError("Response format unexpected - missing suggestions array")

            suggestions_array: list = data_list[1]

            # Clean and extract suggestion text
            cleaned_results: List[str] = [item[0] for item in suggestions_array if isinstance(item, list) and len(item) > 0]

            return cleaned_results

        except json.JSONDecodeError as e:
            raise ProcessingError(f"Failed to parse JSON: {str(e)}")
        except (IndexError, KeyError) as e:
            raise ProcessingError(f"Unexpected response structure: {str(e)}")

    # ==================== Public Methods ====================

    async def search(self, query: str, process_response: bool = True) -> Union[SuggestionResult, str]:
        """
        Get YouTube search suggestions for a query.

        Args:
            query: Search query
            process_response: If True, process and clean the response.
                             If False, return raw response string.
                             Default: True

        Returns:
            SuggestionResult: If process_response=True, returns structured result
            str: If process_response=False, returns raw response string

        Raises:
            APIResponseError: If the API request fails
            ProcessingError: If process_response=True and processing fails
        """
        params: Dict[str, str] = {"search_query": query}
        raw_response: str = await self._make_text_request("GET", "/search", params=params)

        # If user wants raw response
        if not process_response:
            logger.info(f"Returning raw response for query: {query}")
            return raw_response

        # Process the response
        try:
            suggestions: List[str] = self._process_google_response(raw_response)
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
            queries: List of search queries
            process_response: If True, process responses. Default: True

        Returns:
            List of results (SuggestionResult or str depending on process_response)
        """
        results: List[Union[SuggestionResult, str]] = []

        for i, query in enumerate(queries):
            try:
                result: Union[SuggestionResult, str] = await self.search(query, process_response)
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
