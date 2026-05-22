import logging
import re

from Services.llm import LLMService

# ---------------------------------------------------
# Logging Configuration
# ---------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------
# Constants
# ---------------------------------------------------

PROMPT_INJECTION_PATTERNS = [
    "ignore the instructions",
    "ignore instructions",
    "ignore previous instructions",
    "disregard previous instructions",
    "forget previous instructions",
    "override previous instructions",
    "bypass previous instructions",
    "circumvent previous instructions",
    "evade previous instructions",
    "contravene previous instructions",
    "defy previous instructions",
    "violate previous instructions",
]

MALICIOUS_KEYWORDS = [
    "hack",
    "attack",
    "exploit",
    "malware",
    "phishing",
]

PII_PATTERNS = {
    "EMAIL": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "PHONE": r"\b(?:\+91[-\s]?)?[6-9]\d{9}\b",
    "AADHAAR": r"\b\d{4}\s?\d{4}\s?\d{4}\b",
    "PAN": r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b",
    "CREDIT_CARD": r"\b(?:\d[ -]*?){13,16}\b",
    "IP_ADDRESS": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}


class InputGuardrails(LLMService):
    """
    Production-grade Input Guardrails for LLM Applications.

    Features:
    ----------
    1. Input Length Validation
    2. Prompt Injection Detection
    3. Malicious Content Detection
    4. PII Detection & Masking
    5. Fallback Safe Response
    6. Logging & Exception Handling
    """

    def input_length(self, input_text: str, max_length: int = 800) -> str:
        """
        Restrict input length to prevent oversized prompts.

        Args:
            input_text (str): User input text
            max_length (int): Maximum allowed length

        Returns:
            str: Truncated or original text
        """

        try:
            logger.info("Validating input length")

            if len(input_text) > max_length:
                logger.warning(
                    f"Input exceeded max length of {max_length}. Truncating input."
                )
                return input_text[:max_length]

            return input_text

        except Exception as error:
            logger.exception(f"Error during input length validation: {error}")
            raise

    def prompt_injection(self, input_text: str) -> bool:
        """
        Detect prompt injection attempts.

        Args:
            input_text (str): User input

        Returns:
            bool: True if prompt injection detected
        """

        try:
            logger.info("Checking for prompt injection")

            input_text = input_text.lower()

            for pattern in PROMPT_INJECTION_PATTERNS:
                if pattern in input_text:
                    logger.warning(f"Prompt injection detected: {pattern}")
                    return True

            return False

        except Exception as error:
            logger.exception(f"Error during prompt injection check: {error}")
            raise

    def malicious_content(self, input_text: str) -> bool:
        """
        Detect malicious or unsafe keywords.

        Args:
            input_text (str): User input

        Returns:
            bool: True if malicious content detected
        """

        try:
            logger.info("Checking for malicious content")

            input_text = input_text.lower()

            for keyword in MALICIOUS_KEYWORDS:
                if keyword in input_text:
                    logger.warning(f"Malicious keyword detected: {keyword}")
                    return True

            return False

        except Exception as error:
            logger.exception(f"Error during malicious content detection: {error}")
            raise

    def fallback_response(self) -> str:
        """
        Generate safe fallback response for blocked inputs.

        Returns:
            str: Safe assistant response
        """

        try:
            logger.info("Generating fallback response")

            fallback_system_prompt = """
            You are a helpful assistant.
            However, the input provided is not appropriate.
            Respond politely that the request cannot be processed.
            """

            return self.get_response(
                system_prompt=fallback_system_prompt,
                user_input="Malicious user input detected",
            )

        except Exception as error:
            logger.exception(f"Error generating fallback response: {error}")
            return "Input cannot be processed safely."

    def pii_detection_and_masking(self, input_text: str) -> str:
        """
        Detect and mask Personally Identifiable Information (PII).

        Supported PII:
        ----------------
        - Email
        - Phone Number
        - Aadhaar Number
        - PAN Number
        - Credit Card Number
        - IP Address

        Args:
            input_text (str): User input

        Returns:
            str: Masked text
        """

        try:
            logger.info("Starting PII detection and masking")

            detected_pii = {}

            # ---------------------------------------
            # Detect PII
            # ---------------------------------------

            for pii_type, pattern in PII_PATTERNS.items():
                matches = re.findall(pattern, input_text)

                if matches:
                    detected_pii[pii_type] = matches
                    logger.warning(f"{pii_type} detected")

            logger.info(f"Detected PII Types: {list(detected_pii.keys())}")

            # ---------------------------------------
            # Masking Functions
            # ---------------------------------------

            def mask_email(match):
                email = match.group()
                username, domain = email.split("@")

                masked_username = username[:2] + "*" * max(len(username) - 2, 0)

                return masked_username + "@" + domain

            def mask_phone(match):
                phone = match.group()

                return phone[:2] + "XXXXXX" + phone[-2:]

            # ---------------------------------------
            # Apply Masking
            # ---------------------------------------

            masked_text = input_text

            masked_text = re.sub(
                PII_PATTERNS["EMAIL"],
                mask_email,
                masked_text,
            )

            masked_text = re.sub(
                PII_PATTERNS["PHONE"],
                mask_phone,
                masked_text,
            )

            masked_text = re.sub(
                PII_PATTERNS["AADHAAR"],
                "XXXX XXXX XXXX",
                masked_text,
            )

            masked_text = re.sub(
                PII_PATTERNS["PAN"],
                "XXXXX9999X",
                masked_text,
            )

            masked_text = re.sub(
                PII_PATTERNS["CREDIT_CARD"],
                "XXXX-XXXX-XXXX-XXXX",
                masked_text,
            )

            masked_text = re.sub(
                PII_PATTERNS["IP_ADDRESS"],
                "XXX.XXX.XXX.XXX",
                masked_text,
            )

            logger.info("PII masking completed successfully")

            return masked_text

        except Exception as error:
            logger.exception(f"Error during PII masking: {error}")
            raise

    def run_input_guardrails(self, input_text: str):
        """
        Execute all input guardrails sequentially.

        Flow:
        ------
        1. Validate input length
        2. Detect prompt injection
        3. Detect malicious content
        4. Mask PII data

        Args:
            input_text (str): Raw user input

        Returns:
            tuple:
                - Processed/Safe text
                - Boolean indicating whether guardrail triggered
        """

        try:
            logger.info("Running input guardrails")

            input_guardrail_triggered = False

            # ---------------------------------------
            # Input Length Validation
            # ---------------------------------------

            input_text = self.input_length(input_text)

            # ---------------------------------------
            # Prompt Injection Detection
            # ---------------------------------------

            if self.prompt_injection(input_text):
                logger.warning("Prompt injection guardrail triggered")

                input_guardrail_triggered = True

                return (
                    self.fallback_response(),
                    input_guardrail_triggered,
                )

            # ---------------------------------------
            # Malicious Content Detection
            # ---------------------------------------

            if self.malicious_content(input_text):
                logger.warning("Malicious content guardrail triggered")

                input_guardrail_triggered = True

                return (
                    self.fallback_response(),
                    input_guardrail_triggered,
                )

            # ---------------------------------------
            # PII Detection & Masking
            # ---------------------------------------

            input_text = self.pii_detection_and_masking(input_text)

            logger.info("Input guardrails completed successfully")

            return input_text, input_guardrail_triggered

        except Exception as error:
            logger.exception(f"Error while running input guardrails: {error}")

            return (
                "Error occurred while processing input safely.",
                True,
            )