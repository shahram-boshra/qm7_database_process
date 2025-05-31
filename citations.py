# citations.py
import logging
from typing import Dict, Any, List

try:
    from exceptions import (
        InvalidCitationDataError,
        MalformedCitationFieldWarning,
        CitationProcessingError
    )
except ImportError:
    class InvalidCitationDataError(TypeError): pass
    class MalformedCitationFieldWarning(UserWarning): pass
    class CitationProcessingError(ValueError): pass


logger = logging.getLogger(__name__)


def format_citation_for_log(citation_data: Dict[str, Any]) -> str:
    citation_key: str = citation_data.get('key', 'N/A')

    if not isinstance(citation_data, dict):
        raise InvalidCitationDataError(type(citation_data), message=f"Citation key '{citation_key}': citation_data must be a dictionary. Received type: {type(citation_data)}.")

    if "full_citation" in citation_data:
        full_citation_content: Any = citation_data["full_citation"]
        if not isinstance(full_citation_content, str):
            logger.warning(f"Warning: Attempting to convert 'full_citation' to string for key '{citation_key}' as it's not a string.")
            try:
                full_citation_content = str(full_citation_content)
                logger.warning(f"Successfully converted 'full_citation' for key '{citation_key}' to string. Original type: {type(citation_data['full_citation'])}.")
            except Exception as e:
                raise MalformedCitationFieldWarning(
                    citation_key, 'full_citation', full_citation_content,
                    message=f"Failed to convert 'full_citation' to string for key '{citation_key}': {e}",
                    original_exception=e
                ) from e
            
        try:
            indented_lines: List[str] = [f"    {line.strip()}" for line in full_citation_content.strip().split('\n')]
            return "\n".join(indented_lines)
        except AttributeError as e:
            raise MalformedCitationFieldWarning(
                citation_key, 'full_citation', full_citation_content,
                message=f"Error processing 'full_citation' for citation key '{citation_key}'. Value: {full_citation_content}",
                original_exception=e
            ) from e
        except Exception as e:
            raise CitationProcessingError(
                citation_key,
                message=f"An unexpected error occurred while processing 'full_citation' for key '{citation_key}': {e}",
                original_exception=e
            ) from e

    parts: List[str] = []
    
    def get_safe_field(data_dict: Dict[str, Any], field_name: str, prefix: str = "", suffix: str = "") -> str:
        value: Any = data_dict.get(field_name)
        if value is None:
            return ""
        if not isinstance(value, (str, int, float)):
            logger.warning(f"Warning: Citation key '{citation_key}' has '{field_name}' field of unexpected type '{type(value)}'. Attempting to convert to string.")
            try:
                return f"    {prefix}{str(value)}{suffix}"
            except Exception as e:
                raise MalformedCitationFieldWarning(
                    citation_key, field_name, value,
                    message=f"Failed to convert '{field_name}' to string for key '{citation_key}': {e}",
                    original_exception=e
                ) from e
        return f"    {value}{suffix}"
    
    try:
        parts.append(get_safe_field(citation_data, 'authors', suffix="."))
        parts.append(get_safe_field(citation_data, 'title', prefix="\"", suffix="\""))
        parts.append(get_safe_field(citation_data, 'journal', suffix="."))
        parts.append(get_safe_field(citation_data, 'conference', suffix="."))
        
        mid_info: List[str] = []
        for field in ['volume', 'issue', 'pages']:
            value: Any = citation_data.get(field)
            if value is not None:
                if not isinstance(value, (str, int)):
                    logger.warning(f"Warning: Citation key '{citation_key}' has '{field}' field of unexpected type '{type(value)}'. Attempting to convert to string.")
                    try:
                        value = str(value)
                    except Exception as e:
                        raise MalformedCitationFieldWarning(
                            citation_key, field, value,
                            message=f"Failed to convert '{field}' to string for key '{citation_key}': {e}",
                            original_exception=e
                        ) from e

                if value is not None:
                    if field == 'issue':
                        mid_info.append(f"({value})")
                    elif field == 'pages':
                        mid_info.append(f"pages {value}")
                    else:
                        mid_info.append(str(value))

        if mid_info:
            parts.append(f"    {', '.join(mid_info)}.")
            
        parts.append(get_safe_field(citation_data, 'year', suffix="."))
        parts.append(get_safe_field(citation_data, 'doi', prefix="DOI: ", suffix="."))
        parts.append(get_safe_field(citation_data, 'note', suffix="."))
        
        return "\n".join([part for part in parts if part.strip()])
    except MalformedCitationFieldWarning:
        raise
    except Exception as e:
        raise CitationProcessingError(
            citation_key,
            message=f"An unexpected error occurred during fallback citation formatting for key '{citation_key}': {e}",
            original_exception=e
        ) from e


def log_qm7_citations(citations_data: List[Dict[str, str]]) -> None:
    logger.info("\n--- Citations for QM7 Dataset ---")
    if not citations_data:
        logger.warning("No citation data provided. No citations to log.")
        logger.info("----------------------------------\n")
        return

    for i, citation_data in enumerate(citations_data):
        citation_key: str = citation_data.get('key', 'N/A')
        try:
            prefix: str = f"[{i+1}] " if not citation_data.get("full_citation", "").strip().startswith(f"[{i+1}]") else ""
            formatted_string: str = format_citation_for_log(citation_data)
            
            lines: List[str] = formatted_string.split('\n')
            if lines:
                lines[0] = f"    {prefix}{lines[0].lstrip()}"
            else:
                logger.warning(f"Formatting for citation at index {i} (key: {citation_key}) resulted in empty string. Logging placeholder.")
                lines = [f"    [CITATION {i+1}]: No content generated."]

            logger.info("\n".join(lines))
        except (InvalidCitationDataError, MalformedCitationFieldWarning, CitationProcessingError) as e:
            logger.error(f"Error processing citation entry (index {i}, key: {citation_key}): {e}")
            logger.info(f"    [UNLOGGED ENTRY {i+1}]: Skipping due to formatting error.")
            if hasattr(e, 'original_exception') and e.original_exception:
                logger.debug(f"Original exception for {citation_key}: {e.original_exception}", exc_info=True)
            continue
        except Exception as e:
            logger.critical(f"An unhandled critical error occurred while processing citation at index {i} (key: {citation_key}): {e}", exc_info=True)
            logger.info(f"    [FATAL ERROR] Could not process citation {i+1}. Skipping.")
    logger.info("----------------------------------\n")
