class PromptSecurityFilter:
    """
    Enhanced security filter with comprehensive protection against various attacks
    """
    
    def __init__(self):
        """Initialize security filter with comprehensive protection patterns"""
        # Compile regex patterns once for efficiency
        self.compiled_patterns = [
            # Original patterns (from your existing code)
            re.compile(r"\b(sh|bash|zsh|ksh|fish)\b", re.IGNORECASE),  # Shell names
            re.compile(r"\b(exec|eval|os\.system|subprocess|popen|run)\b", re.IGNORECASE),  # Python exec calls
            re.compile(r"\|s*\w+", re.IGNORECASE),  # Command chaining
            re.compile(r"[|]+", re.IGNORECASE),  # Pipe operator
            re.compile(r"[^>]+", re.IGNORECASE),  # Backtick execution
            re.compile(r"\$\(.*\)", re.IGNORECASE),  # Subshell execution
            re.compile(r"rm\s+-rf", re.IGNORECASE),  # Dangerous commands
            re.compile(r"cat\s+/etc/passwd", re.IGNORECASE),  # Sensitive file access
            
            # NEW: Enhanced security patterns
            re.compile(r"(ignore|disregard|forget).*(previous|above|instruction)", re.IGNORECASE),  # Prompt injection
            re.compile(r"<script[^>]*>.*?</script>", re.IGNORECASE | re.DOTALL),  # XSS attacks
            re.compile(r"javascript:", re.IGNORECASE),  # JavaScript injection
            re.compile(r"\.\.\/|\.\.\\", re.IGNORECASE),  # Path traversal
            re.compile(r"(token|key|password|secret)[\s:=][A-Za-z0-9+/]{10,}", re.IGNORECASE),  # Credential leaks
            re.compile(r"(act\s+as|pretend\s+to\s+be|roleplay\s+as)", re.IGNORECASE),  # Role manipulation
            re.compile(r"(system|assistant|user):", re.IGNORECASE),  # Role injection
        ]
        
        # Track filtering statistics
        self.filtered_count = 0
        self.suspicious_attempts = 0
    
    def sanitize(self, text: str) -> str:
        """
        Sanitize input text with improved safety checks
        
        Args:
            text: Input text to sanitize
            
        Returns:
            Sanitized text safe for processing
        """
        # Input validation
        if not isinstance(text, str):
            logging.warning(f"Non-string input to sanitize: {type(text)}")
            text = str(text)
        
        if not text.strip():
            return ""
        
        original_text = text
        original_length = len(text)
        
        # Apply pattern filtering
        for i, pattern in enumerate(self.compiled_patterns):
            old_text = text
            text = pattern.sub("", text)
            if text != old_text:
                logging.debug(f"Security pattern {i} triggered and filtered")
        
        # Apply delimiter filtering (from your original code)
        dangerous_delimiters = ["<", ">", "[", "]", "```system", "```assistant"]
        for delimiter in dangerous_delimiters:
            text = text.replace(delimiter, "")
        
        # Length protection
        max_length = 50000  # Reasonable limit for LLM input
        if len(text) > max_length:
            logging.warning(f"Text truncated from {len(text)} to {max_length} characters")
            text = text[:max_length]
        
        # Remove null bytes and dangerous control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        
        # Log filtering activity
        if text != original_text:
            self.filtered_count += 1
            reduction_percent = ((original_length - len(text)) / original_length * 100) if original_length > 0 else 0
            logging.info(f"Security filter applied (#{self.filtered_count}). Text reduced by {reduction_percent:.1f}%")
        
        return text.strip()
    
    def is_suspicious(self, text: str) -> tuple[bool, list[str]]:
        """
        Check if text contains suspicious patterns without filtering
        
        Args:
            text: Text to analyze
            
        Returns:
            Tuple of (is_suspicious, list_of_reasons)
        """
        if not isinstance(text, str):
            return True, ["Non-string input"]
        
        suspicious_reasons = []
        
        # Check each pattern
        for i, pattern in enumerate(self.compiled_patterns):
            if pattern.search(text):
                suspicious_reasons.append(f"Matched security pattern {i}")
        
        # Check for excessive special characters
        if len(text) > 0:
            special_char_ratio = sum(1 for c in text if not c.isalnum() and not c.isspace()) / len(text)
            if special_char_ratio > 0.5:
                suspicious_reasons.append(f"High special character ratio: {special_char_ratio:.2f}")
        
        # Check for excessive length
        if len(text) > 100000:
            suspicious_reasons.append("Excessive length")
        
        # Check encoding
        try:
            text.encode('utf-8')
        except UnicodeEncodeError:
            suspicious_reasons.append("Invalid Unicode encoding")
        
        is_suspicious = len(suspicious_reasons) > 0
        
        if is_suspicious:
            self.suspicious_attempts += 1
            logging.warning(f"Suspicious input detected (#{self.suspicious_attempts}): {suspicious_reasons}")
        
        return is_suspicious, suspicious_reasons
    
    def get_stats(self) -> dict:
        """Get filtering statistics"""
        return {
            "total_filtered": self.filtered_count,
            "suspicious_attempts": self.suspicious_attempts
        }
    
    def reset_stats(self):
        """Reset filtering statistics"""
        self.filtered_count = 0
        self.suspicious_attempts = 0
        logging.info("Security filter statistics reset")
