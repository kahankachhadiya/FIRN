"""Qwen-based metadata extraction module for documents.

This module provides metadata extraction functionality using Qwen3 VL 30B A3B
via LM Studio API. It implements the Bibliographic Metadata Extractor prompt
for extracting structured metadata from documents.

Requirements:
- 5.1: Extract title from document header or first heading
- 5.2: Clean author names (remove affiliation markers)
- 5.3: Extract dates in YYYY-MM-DD format
- 5.4: Extract language as ISO 639-1 code
- 5.5: Generate 5-10 semantic keywords from Abstract and Introduction
- 11.3: Use Bibliographic Metadata Extractor prompt
"""

import re
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime

from external_clients.openai_compatible_client import OpenAICompatibleClient as LMStudioClient
from extractors.config import PromptsConfig


@dataclass
class DocumentMetadata:
    """Represents extracted document metadata."""
    title: str
    authors: List[str]
    created_date: str
    language: str
    tags: List[str]
    extra_metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metadata to dictionary."""
        return {
            "title": self.title,
            "authors": self.authors,
            "created_date": self.created_date,
            "language": self.language,
            "tags": self.tags,
            "extra_metadata": self.extra_metadata
        }


class LLMMetadataExtractor:
    """Metadata extraction using Qwen via LM Studio.
    
    Implements the Bibliographic Metadata Extractor prompt for extracting
    precise metadata fields from research papers and documents.
    
    Features:
    - Title extraction with cleaning
    - Author name cleaning (remove affiliation markers)
    - Date extraction and formatting (YYYY-MM-DD)
    - Language detection (ISO 639-1 codes)
    - Semantic keyword generation
    - Extra metadata (journal, DOI, affiliations)
    """
    
    def __init__(
        self,
        lm_client: Optional[LMStudioClient] = None,
        prompts_config: Optional[PromptsConfig] = None,
    ):
        """
        Initialize LLMMetadataExtractor.
        
        Args:
            lm_client: LM Studio client instance (creates new if None)
            prompts_config: Prompts configuration (uses default if None)
        """
        self.lm_client = lm_client or LMStudioClient()
        self.prompts_config = prompts_config or PromptsConfig()
        
        print("LLMMetadataExtractor: Initialized")
    
    def extract_metadata(
        self,
        markdown: str,
        filename: str = "",
        file_path: str = "",
        source_type: str = "document"
    ) -> Dict[str, Any]:
        """
        Extract metadata from markdown content using Qwen.
        
        Implements Requirements:
        - 5.1: Extract title from document header or first heading
        - 5.2: Clean author names (remove affiliation markers)
        - 5.3: Extract dates in YYYY-MM-DD format
        - 5.4: Extract language as ISO 639-1 code
        - 5.5: Generate 5-10 semantic keywords
        - 11.3: Use Bibliographic Metadata Extractor prompt
        
        Args:
            markdown: Markdown content to extract metadata from
            filename: Original filename (for fallback title)
            file_path: Full file path
            source_type: Type of source ("document", "audio", "image")
            
        Returns:
            Dictionary with document_metadata fields
        """
        if not markdown or not markdown.strip():
            raise ValueError(
                "LLMMetadataExtractor: Empty markdown provided. "
                "Cannot extract metadata from empty content."
            )
        
        # Extract metadata via LLM
        metadata = self._extract_with_llm(markdown)
        
        # Post-process and validate metadata
        metadata = self._post_process_metadata(
            metadata, filename, file_path, source_type
        )
        
        print(f"LLMMetadataExtractor: Extracted metadata for '{metadata.get('title', 'Unknown')}'")
        return metadata
    
    def _extract_with_llm(self, markdown: str) -> Dict[str, Any]:
        """
        Use Qwen LLM to extract metadata.
        
        Args:
            markdown: Markdown content (first 3000 chars for efficiency)
            
        Returns:
            Metadata dictionary
        """
        # Use first 3000 chars (header/abstract area)
        content_for_metadata = markdown[:3000]
        
        prompt = self.prompts_config.metadata_prompt
        
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": content_for_metadata}
        ]
        
        try:
            response = self.lm_client.chat_completion(
                messages=messages,
                max_tokens=1000,
                temperature=0.1,  # Low temperature for consistent extraction
                json_mode=True
            )
            
            if not response:
                raise RuntimeError("LLMMetadataExtractor: Empty response from LLM.")

            # Log raw response for debugging
            print(f"LLMMetadataExtractor: Raw LLM response:\n{response}\n--- end raw response ---")

            # Strip markdown code fences if the LLM wrapped the JSON
            cleaned = response.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r'^```(?:json)?\s*', '', cleaned)
                cleaned = re.sub(r'\s*```$', '', cleaned)
                cleaned = cleaned.strip()

            # Parse JSON response — try strict first, then with repair
            result = None
            try:
                result = json.loads(cleaned)
            except json.JSONDecodeError:
                # Attempt lightweight repair: truncate at last valid closing brace
                last_brace = cleaned.rfind("}")
                if last_brace != -1:
                    try:
                        result = json.loads(cleaned[:last_brace + 1])
                        print("LLMMetadataExtractor: JSON repaired by truncation")
                    except json.JSONDecodeError:
                        pass

            if result is None:
                # LLM returned unparseable JSON — use rule-based fallback
                print("LLMMetadataExtractor: JSON parse failed, using rule-based fallback")
                return self._fallback_extract(response)

            metadata = result.get("document_metadata", {})
            
            if not metadata:
                raise RuntimeError("LLMMetadataExtractor: LLM response contained no document_metadata.")
            
            return metadata
            
        except json.JSONDecodeError as e:
            print(f"LLMMetadataExtractor: JSON parse error ({e}), using rule-based fallback")
            return self._fallback_extract(markdown)
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"LLMMetadataExtractor: LLM extraction error: {e}") from e
    
    def _fallback_extract(self, markdown: str, filename: str = "") -> Dict[str, Any]:
        """
        Rule-based fallback metadata extraction when LLM fails.
        
        Implements basic extraction with:
        - Title from first heading or filename
        - Author extraction with cleaning
        - Date extraction
        - Language detection
        - Simple keyword extraction
        
        Args:
            markdown: Markdown content
            filename: Filename for fallback title
            
        Returns:
            Metadata dictionary
        """
        metadata = {
            "title": "",
            "authors": [],
            "created_date": "",
            "language": "en",
            "tags": [],
            "extra_metadata": {}
        }
        
        # Extract title (Requirement 5.1)
        metadata["title"] = self._extract_title(markdown, filename)
        
        # Extract authors (Requirement 5.2)
        metadata["authors"] = self._extract_authors(markdown)
        
        # Extract date (Requirement 5.3)
        metadata["created_date"] = self._extract_date(markdown)
        
        # Detect language (Requirement 5.4)
        metadata["language"] = self._detect_language(markdown)
        
        # Extract keywords (Requirement 5.5)
        metadata["tags"] = self._extract_keywords(markdown)
        
        # Extract extra metadata
        metadata["extra_metadata"] = self._extract_extra_metadata(markdown)
        
        return metadata
    
    def _extract_title(self, markdown: str, filename: str = "") -> str:
        """
        Extract title from document header or first heading.
        
        Implements Requirement 5.1: Extract title from document header
        or first heading.
        
        Args:
            markdown: Markdown content
            filename: Fallback filename
            
        Returns:
            Document title
        """
        lines = markdown.split('\n')
        
        # Look for first heading
        for line in lines[:20]:  # Check first 20 lines
            # Match markdown headings
            heading_match = re.match(r'^#{1,3}\s+(.+)$', line)
            if heading_match:
                title = heading_match.group(1).strip()
                # Clean trailing asterisks or footnote markers
                title = re.sub(r'[*†‡§¶#]+$', '', title).strip()
                if title:
                    return title
            
            # Look for title-like patterns (all caps, bold, etc.)
            # Only consider lines that look like titles (short, capitalized, not sentences)
            if line.strip() and 10 < len(line.strip()) < 100:
                # Check if line looks like a title (not too long, capitalized, no period at end)
                if line.strip()[0].isupper() and not line.strip().endswith('.'):
                    # Not a heading but could be title
                    if not re.match(r'^(Abstract|Introduction|Background)', line, re.IGNORECASE):
                        title = line.strip()
                        title = re.sub(r'[*†‡§¶#]+$', '', title).strip()
                        if title:
                            return title
        
        # Fallback to filename
        if filename:
            # Remove extension and clean up
            title = re.sub(r'\.[^.]+$', '', filename)
            title = title.replace('_', ' ').replace('-', ' ')
            return title.strip()
        
        return "Untitled Document"
    
    def _extract_authors(self, markdown: str) -> List[str]:
        """
        Extract author names and clean affiliation markers.
        
        Implements Requirement 5.2: Extract author names and remove
        affiliation markers (numbers, superscripts, symbols).
        
        Args:
            markdown: Markdown content
            
        Returns:
            List of cleaned author names
        """
        authors = []
        lines = markdown.split('\n')
        
        # Look for author patterns in first 30 lines
        for i, line in enumerate(lines[:30]):
            # Skip headings and empty lines
            if re.match(r'^#{1,6}\s+', line) or not line.strip():
                continue
            
            # Look for "Author:" or "Authors:" patterns
            author_match = re.match(r'^Authors?:\s*(.+)$', line, re.IGNORECASE)
            if author_match:
                author_text = author_match.group(1)
                authors = self._parse_author_list(author_text)
                break
            
            # Look for "By:" patterns
            by_match = re.match(r'^By:\s*(.+)$', line, re.IGNORECASE)
            if by_match:
                author_text = by_match.group(1)
                authors = self._parse_author_list(author_text)
                break
            
            # Look for comma-separated names after title
            # (common pattern: Title\nAuthor1, Author2, Author3)
            if i > 0 and ',' in line and len(line) < 200:
                # Check if previous line looks like a title
                prev_line = lines[i-1].strip()
                if prev_line and len(prev_line) > 10:
                    # This might be author list
                    potential_authors = self._parse_author_list(line)
                    if potential_authors:
                        authors = potential_authors
                        break
        
        return authors
    
    def _parse_author_list(self, author_text: str) -> List[str]:
        """
        Parse and clean author list.
        
        Removes affiliation markers: numbers, superscripts, symbols.
        
        Args:
            author_text: Raw author text
            
        Returns:
            List of cleaned author names
        """
        # Split by common separators
        if ',' in author_text and ' and ' in author_text:
            # Mixed format: "Author1, Author2, and Author3"
            author_text = author_text.replace(' and ', ', ')
        elif ' and ' in author_text:
            # "Author1 and Author2"
            author_text = author_text.replace(' and ', ', ')
        
        # Split by comma or semicolon
        raw_authors = re.split(r'[,;]', author_text)
        
        cleaned_authors = []
        for author in raw_authors:
            # Remove affiliation markers
            # Remove superscript numbers: ¹²³⁴⁵⁶⁷⁸⁹⁰
            author = re.sub(r'[¹²³⁴⁵⁶⁷⁸⁹⁰]+', '', author)
            # Remove regular numbers and symbols
            author = re.sub(r'[0-9*†‡§¶#]+', '', author)
            # Remove parenthetical affiliations
            author = re.sub(r'\([^)]+\)', '', author)
            # Remove email addresses
            author = re.sub(r'[\w\.-]+@[\w\.-]+', '', author)
            
            # Clean whitespace
            author = author.strip()
            
            # Validate: should have at least 2 characters and look like a name
            if len(author) >= 2 and re.search(r'[A-Za-z]', author):
                cleaned_authors.append(author)
        
        return cleaned_authors
    
    def _extract_date(self, markdown: str) -> str:
        """
        Extract publication/creation date.
        
        Implements Requirement 5.3: Look for "Published," "Accepted,"
        or "Received" dates. Format: YYYY-MM-DD.
        
        Args:
            markdown: Markdown content
            
        Returns:
            Date in YYYY-MM-DD format or empty string
        """
        lines = markdown.split('\n')
        
        # Look for date patterns in first 50 lines
        for line in lines[:50]:
            # Look for explicit date labels
            date_match = re.search(
                r'(Published|Accepted|Received|Date|Created):\s*([^\n]+)',
                line,
                re.IGNORECASE
            )
            if date_match:
                date_text = date_match.group(2).strip()
                formatted_date = self._parse_date(date_text)
                if formatted_date:
                    return formatted_date
            
            # Look for standalone date patterns
            # YYYY-MM-DD format
            date_match = re.search(r'\b(\d{4})-(\d{2})-(\d{2})\b', line)
            if date_match:
                return date_match.group(0)
            
            # Month DD, YYYY format
            date_match = re.search(
                r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})\b',
                line,
                re.IGNORECASE
            )
            if date_match:
                formatted_date = self._parse_date(date_match.group(0))
                if formatted_date:
                    return formatted_date
        
        return ""
    
    def _parse_date(self, date_text: str) -> str:
        """
        Parse various date formats to YYYY-MM-DD.
        
        Args:
            date_text: Raw date text
            
        Returns:
            Date in YYYY-MM-DD format or empty string
        """
        # Try common date formats
        formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%B %d, %Y',
            '%b %d, %Y',
            '%d %B %Y',
            '%d %b %Y',
            '%m/%d/%Y',
            '%d/%m/%Y',
        ]
        
        date_text = date_text.strip()
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_text, fmt)
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        # Try to extract just year if full date fails
        year_match = re.search(r'\b(19|20)\d{2}\b', date_text)
        if year_match:
            return f"{year_match.group(0)}-01-01"
        
        return ""
    
    def _detect_language(self, markdown: str) -> str:
        """
        Detect document language.
        
        Implements Requirement 5.4: Detect language, return ISO 639-1 code.
        
        Args:
            markdown: Markdown content
            
        Returns:
            ISO 639-1 language code (default: "en")
        """
        # Simple heuristic-based detection
        # Check for common words in different languages
        
        text_sample = markdown[:1000].lower()
        
        # English indicators
        en_words = ['the', 'and', 'is', 'are', 'was', 'were', 'this', 'that']
        en_count = sum(1 for word in en_words if f' {word} ' in text_sample)
        
        # Spanish indicators
        es_words = ['el', 'la', 'los', 'las', 'de', 'que', 'es', 'en']
        es_count = sum(1 for word in es_words if f' {word} ' in text_sample)
        
        # French indicators
        fr_words = ['le', 'la', 'les', 'de', 'et', 'est', 'dans', 'que']
        fr_count = sum(1 for word in fr_words if f' {word} ' in text_sample)
        
        # German indicators
        de_words = ['der', 'die', 'das', 'und', 'ist', 'sind', 'ein', 'eine']
        de_count = sum(1 for word in de_words if f' {word} ' in text_sample)
        
        # Determine language by highest count
        counts = {
            'en': en_count,
            'es': es_count,
            'fr': fr_count,
            'de': de_count,
        }
        
        max_lang = max(counts, key=counts.get)
        
        # Require at least 3 matches to be confident
        if counts[max_lang] >= 3:
            return max_lang
        
        # Default to English
        return 'en'
    
    def _extract_keywords(self, markdown: str) -> List[str]:
        """
        Extract semantic keywords from content.
        
        Implements Requirement 5.5: Generate 5-10 semantic keywords
        from Abstract and Introduction.
        
        Args:
            markdown: Markdown content
            
        Returns:
            List of 5-10 keywords
        """
        keywords = []
        
        # Extract Abstract and Introduction sections
        abstract = self._extract_section(markdown, 'Abstract')
        intro = self._extract_section(markdown, 'Introduction')
        
        content = f"{abstract} {intro}"
        
        if not content.strip():
            # Use first 1000 chars if no abstract/intro found
            content = markdown[:1000]
        
        # Simple keyword extraction: find capitalized terms and noun phrases
        # Remove common words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'we', 'our', 'their', 'its', 'it', 'they', 'them'
        }
        
        # Find capitalized words (potential keywords)
        words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        
        # Count frequency
        word_freq = {}
        for word in words:
            word_lower = word.lower()
            if word_lower not in stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and take top keywords
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, freq in sorted_keywords[:10]]
        
        # If we don't have enough, add some common technical terms
        if len(keywords) < 5:
            # Extract any words that appear multiple times
            all_words = re.findall(r'\b[a-z]{4,}\b', content.lower())
            word_freq = {}
            for word in all_words:
                if word not in stop_words:
                    word_freq[word] = word_freq.get(word, 0) + 1
            
            # Add high-frequency words
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            for word, freq in sorted_words:
                if freq >= 3 and word not in [k.lower() for k in keywords]:
                    keywords.append(word)
                    if len(keywords) >= 10:
                        break
        
        return keywords[:10]
    
    def _extract_section(self, markdown: str, section_name: str) -> str:
        """
        Extract a specific section from markdown.
        
        Args:
            markdown: Markdown content
            section_name: Section heading to find
            
        Returns:
            Section content
        """
        lines = markdown.split('\n')
        section_content = []
        in_section = False
        
        for i, line in enumerate(lines):
            # Check if this is the target section heading
            # Use word boundary to match section name
            pattern = r'^#{1,6}\s+' + re.escape(section_name) + r'\b'
            match = re.match(pattern, line, re.IGNORECASE)
            if match:
                in_section = True
                continue
            
            # Check if we hit another section heading
            if in_section and re.match(r'^#{1,6}\s+', line):
                break
            
            if in_section:
                section_content.append(line)
        
        return '\n'.join(section_content)
    
    def _extract_extra_metadata(self, markdown: str) -> Dict[str, Any]:
        """
        Extract additional metadata fields.
        
        Extracts journal name, DOI, and affiliations if available.
        
        Args:
            markdown: Markdown content
            
        Returns:
            Dictionary with extra metadata
        """
        extra = {}
        
        lines = markdown.split('\n')
        
        # Look for DOI
        for line in lines[:50]:
            doi_match = re.search(r'DOI:\s*(10\.\d+/[^\s]+)', line, re.IGNORECASE)
            if doi_match:
                extra['doi'] = doi_match.group(1)
                break
            
            # Alternative DOI format
            doi_match = re.search(r'\b(10\.\d+/[^\s]+)\b', line)
            if doi_match and 'doi' not in extra:
                extra['doi'] = doi_match.group(1)
        
        # Look for journal name
        for line in lines[:30]:
            journal_match = re.search(
                r'(Journal|Proceedings|Conference):\s*([^\n]+)',
                line,
                re.IGNORECASE
            )
            if journal_match:
                extra['journal'] = journal_match.group(2).strip()
                break
        
        # Look for affiliations
        affiliations = []
        for line in lines[:50]:
            # Look for university/institution patterns
            if re.search(r'University|Institute|Laboratory|College|Department', line, re.IGNORECASE):
                # Clean up the line
                affiliation = re.sub(r'^[0-9*†‡§¶#]+', '', line).strip()
                if affiliation and len(affiliation) > 10:
                    affiliations.append(affiliation)
        
        if affiliations:
            extra['affiliations'] = ', '.join(affiliations[:3])  # Limit to 3
        
        return extra
    
    def _get_fallback_metadata(
        self, filename: str, file_path: str, source_type: str
    ) -> Dict[str, Any]:
        """
        Get minimal fallback metadata when extraction fails.
        
        Args:
            filename: Filename
            file_path: Full file path
            source_type: Source type
            
        Returns:
            Minimal metadata dictionary
        """
        title = filename if filename else "Untitled Document"
        title = re.sub(r'\.[^.]+$', '', title)
        title = title.replace('_', ' ').replace('-', ' ')
        
        return {
            "title": title.strip(),
            "authors": [],
            "created_date": "",
            "language": "en",
            "tags": [],
            "extra_metadata": {}
        }
    
    def _post_process_metadata(
        self,
        metadata: Dict[str, Any],
        filename: str,
        file_path: str,
        source_type: str
    ) -> Dict[str, Any]:
        """
        Post-process and validate metadata.
        
        Ensures all required fields are present and properly formatted.
        
        Args:
            metadata: Raw metadata dictionary
            filename: Filename for fallback
            file_path: Full file path
            source_type: Source type
            
        Returns:
            Validated metadata dictionary
        """
        # Ensure all required fields exist
        processed = {
            "title": metadata.get("title", ""),
            "authors": metadata.get("authors", []),
            "created_date": metadata.get("created_date", ""),
            "language": metadata.get("language", "en"),
            "tags": metadata.get("tags", []),
            "extra_metadata": metadata.get("extra_metadata", {})
        }
        
        # Validate and fix title
        if not processed["title"] or processed["title"] == "Untitled Document":
            processed["title"] = self._extract_title("", filename)
        
        # Clean title
        processed["title"] = re.sub(r'[*†‡§¶#]+$', '', processed["title"]).strip()
        
        # Ensure authors is a list
        if isinstance(processed["authors"], str):
            processed["authors"] = [processed["authors"]]
        elif not isinstance(processed["authors"], list):
            processed["authors"] = []
        
        # Clean author names
        cleaned_authors = []
        for author in processed["authors"]:
            if isinstance(author, str):
                # Remove affiliation markers
                author = re.sub(r'[¹²³⁴⁵⁶⁷⁸⁹⁰0-9*†‡§¶#]+', '', author)
                author = author.strip()
                if author:
                    cleaned_authors.append(author)
        processed["authors"] = cleaned_authors
        
        # Validate date format
        if processed["created_date"]:
            # Ensure YYYY-MM-DD format
            if not re.match(r'^\d{4}-\d{2}-\d{2}$', processed["created_date"]):
                # Try to parse and reformat
                reformatted = self._parse_date(processed["created_date"])
                processed["created_date"] = reformatted
        
        # Validate language code (should be 2 letters)
        if not processed["language"] or len(processed["language"]) != 2:
            processed["language"] = "en"
        
        # Ensure tags is a list and limit to 10
        if isinstance(processed["tags"], str):
            processed["tags"] = [processed["tags"]]
        elif not isinstance(processed["tags"], list):
            processed["tags"] = []
        
        # Clean and limit tags
        cleaned_tags = []
        for tag in processed["tags"][:10]:
            if isinstance(tag, str):
                tag = tag.strip().lower()
                if tag and tag not in cleaned_tags:
                    cleaned_tags.append(tag)
        processed["tags"] = cleaned_tags
        
        # Ensure extra_metadata is a dict
        if not isinstance(processed["extra_metadata"], dict):
            processed["extra_metadata"] = {}
        
        return processed
    
    def get_metadata_json(
        self,
        markdown: str,
        filename: str = "",
        file_path: str = "",
        source_type: str = "document"
    ) -> str:
        """
        Get metadata as JSON string.
        
        Args:
            markdown: Markdown content
            filename: Filename
            file_path: Full file path
            source_type: Source type
            
        Returns:
            JSON string with document_metadata
        """
        metadata = self.extract_metadata(markdown, filename, file_path, source_type)
        return json.dumps({"document_metadata": metadata}, indent=2)


def create_metadata_extractor(
    lm_client: Optional[LMStudioClient] = None,
    prompts_config: Optional[PromptsConfig] = None,
) -> LLMMetadataExtractor:
    """
    Factory function to create a LLMMetadataExtractor instance.
    
    Args:
        lm_client: Optional LM Studio client
        prompts_config: Optional prompts configuration
        
    Returns:
        Configured LLMMetadataExtractor instance
    """
    return LLMMetadataExtractor(
        lm_client=lm_client,
        prompts_config=prompts_config,
    )
