"""Simple authentication manager with optional email token flows."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import smtplib
import secrets
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml


DEFAULT_HASH_ITERATIONS = 180_000
TOKEN_TTL_MINUTES = 60


@dataclass
class UserRecord:
    username: str
    name: str
    email: str
    salt: str
    password_hash: str
    active: bool = False

    @property
    def has_credentials(self) -> bool:
        return bool(self.salt and self.password_hash)


@dataclass
class TokenDispatchResult:
    sent: bool
    token: str
    message: str


class EmailNotifier:
    """Utility wrapper around ``smtplib`` for optional email delivery."""

    def __init__(self, config: Dict[str, object]) -> None:
        self._config = config or {}

    def _resolve(self, key: str) -> Optional[str]:
        env_key = self._config.get(f"{key}_env")
        if env_key:
            env_value = os.environ.get(str(env_key))
            if env_value:
                return env_value
        value = self._config.get(key)
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            return value
        return str(value)

    def _resolve_password(self) -> Optional[str]:
        password_env = self._config.get("password_env")
        if password_env:
            password = os.environ.get(str(password_env))
            if password:
                return password
        return self._resolve("password")

    @property
    def enabled(self) -> bool:
        host = self._resolve("host")
        from_email = self._resolve("from_email")
        if not host or not from_email:
            return False
        username = self._resolve("username")
        if not username:
            return True
        return bool(self._resolve_password())

    def send_token(self, *, to_address: str, subject: str, body: str) -> bool:
        if not self.enabled:
            return False
        host = self._resolve("host")
        from_email = self._resolve("from_email")
        port_value = self._resolve("port")
        if port_value is None:
            port_value = self._config.get("port", 587)
        try:
            port = int(port_value)
        except (ValueError, TypeError):
            port = 587
        username = self._resolve("username")
        password = self._resolve_password()
        if username and not password:
            return False

        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = from_email
        msg["To"] = to_address
        msg.set_content(body)

        try:
            with smtplib.SMTP(host, port, timeout=10) as smtp:
                if bool(self._config.get("starttls", True)):
                    smtp.starttls()
                if username:
                    smtp.login(str(username), str(password))
                smtp.send_message(msg)
            return True
        except Exception:
            return False


class AuthManager:
    """File-backed credential store with activation/reset token workflows."""

    def __init__(self, config_path: Path) -> None:
        self.config_path = config_path
        self._data: Dict[str, object] = {}
        self._load()

    # ------------------------------------------------------------------ loading
    def _load(self) -> None:
        if self.config_path.exists():
            with self.config_path.open("r", encoding="utf-8") as fh:
                self._data = yaml.safe_load(fh) or {}
        else:
            self._data = {}

        self._data.setdefault("users", [])
        self._data.setdefault("allowed_usernames", [])
        self._data.setdefault("allowed_emails", [])
        self._data.setdefault("pending_tokens", {"activation": {}, "reset": {}})
        self._data.setdefault("settings", {"hash_iterations": DEFAULT_HASH_ITERATIONS})
        self._data.setdefault("smtp", {})
        self._purge_expired_tokens()
        self._save()

    def _save(self) -> None:
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with self.config_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(self._data, fh, sort_keys=False)

    # ------------------------------------------------------------------ helpers
    @property
    def _users(self) -> List[Dict[str, object]]:
        return self._data["users"]

    @property
    def _token_store(self) -> Dict[str, Dict[str, Dict[str, object]]]:
        return self._data["pending_tokens"]

    @property
    def _hash_iterations(self) -> int:
        settings = self._data.get("settings") or {}
        return int(settings.get("hash_iterations", DEFAULT_HASH_ITERATIONS))

    def _find_user(self, username: str) -> Optional[Dict[str, object]]:
        username_lower = username.strip().lower()
        for record in self._users:
            if str(record.get("username", "")).lower() == username_lower:
                return record
        return None

    def _find_user_by_email(self, email: str) -> Optional[Dict[str, object]]:
        email_lower = email.strip().lower()
        for record in self._users:
            if str(record.get("email", "")).strip().lower() == email_lower:
                return record
        return None

    def _username_taken(self, username: str) -> bool:
        return self._find_user(username) is not None

    def _hash_password(self, password: str, salt_hex: Optional[str] = None) -> Tuple[str, str]:
        if salt_hex:
            salt = bytes.fromhex(salt_hex)
        else:
            salt = os.urandom(16)
        digest = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            self._hash_iterations,
        )
        return salt.hex(), base64.b64encode(digest).decode("utf-8")

    def _verify_password(self, record: Dict[str, object], password: str) -> bool:
        if not record.get("salt") or not record.get("password_hash"):
            return False
        salt = str(record["salt"])
        expected = str(record["password_hash"])
        _salt, hashed = self._hash_password(password, salt_hex=salt)
        return hmac.compare_digest(expected, hashed)

    def _allowed_username(self, username: str) -> bool:
        allowed = self._data.get("allowed_usernames") or []
        if not allowed:
            return True
        return username.strip().lower() in {str(item).lower() for item in allowed}

    def _allowed_email(self, email: str) -> bool:
        allowed = self._data.get("allowed_emails") or []
        if not allowed:
            return True
        email_lower = email.strip().lower()
        return email_lower in {str(item).lower() for item in allowed}

    def _purge_expired_tokens(self) -> None:
        now = datetime.now(timezone.utc)
        for bucket in ("activation", "reset"):
            entries = self._token_store.setdefault(bucket, {})
            expired = []
            for token, payload in entries.items():
                expires_at = payload.get("expires_at")
                if not expires_at:
                    continue
                try:
                    expiry_dt = datetime.fromisoformat(str(expires_at))
                except ValueError:
                    expired.append(token)
                    continue
                if expiry_dt <= now:
                    expired.append(token)
            for token in expired:
                entries.pop(token, None)
        self._save()

    def _issue_token(self, bucket: str, payload: Dict[str, object]) -> str:
        token = secrets.token_urlsafe(32)
        expires_at = datetime.now(timezone.utc) + timedelta(minutes=TOKEN_TTL_MINUTES)
        payload = dict(payload)
        payload["expires_at"] = expires_at.isoformat()
        self._token_store.setdefault(bucket, {})[token] = payload
        self._save()
        return token

    def _consume_token(self, bucket: str, token: str) -> Optional[Dict[str, object]]:
        entries = self._token_store.setdefault(bucket, {})
        payload = entries.pop(token, None)
        self._save()
        return payload

    # ---------------------------------------------------------------- authentication
    def authenticate(self, username: str, password: str) -> Optional[UserRecord]:
        record = self._find_user(username)
        if not record or not record.get("active"):
            return None
        if not self._verify_password(record, password):
            return None
        return UserRecord(
            username=str(record["username"]),
            name=str(record.get("name") or record["username"]),
            email=str(record.get("email") or ""),
            salt=str(record.get("salt") or ""),
            password_hash=str(record.get("password_hash") or ""),
            active=bool(record.get("active", True)),
        )

    # ----------------------------------------------------------- activation & reset
    def request_activation(self, username: str, email: str, notifier: Optional[EmailNotifier] = None) -> TokenDispatchResult:
        username = username.strip()
        email = email.strip()
        record = self._find_user(username)
        if record is None:
            return TokenDispatchResult(False, "", "The supplied username is not permitted. Contact an administrator.")
        if record.get("active"):
            return TokenDispatchResult(False, "", "Account is already activated.")
        if str(record.get("email", "")).strip().lower() != email.lower():
            return TokenDispatchResult(False, "", "Email does not match our records.")
        if not self._allowed_username(username) or not self._allowed_email(email):
            return TokenDispatchResult(False, "", "This account is not authorized for activation.")

        token = self._issue_token(
            "activation",
            {
                "username": username,
                "email": email,
            },
        )
        message = (
            "Activation token generated. Enter this code in the app within one hour to complete setup."
        )
        return TokenDispatchResult(True, token, message)

    def request_activation_by_email(self, email: str, notifier: Optional[EmailNotifier] = None) -> TokenDispatchResult:
        """Issue an activation token using only an allowed email, without sending email notifications.

        With this flow, the user chooses their username during activation.
        """
        email = email.strip()
        if not self._allowed_email(email):
            return TokenDispatchResult(False, "", "This email is not authorized for activation.")
        existing = self._find_user_by_email(email)
        if existing and existing.get("active"):
            return TokenDispatchResult(False, "", "An active account already exists for this email.")
        token = self._issue_token("activation", {"email": email})
        message = (
            "Activation token generated. Enter this code in the app within one hour to choose your username and password."
        )
        return TokenDispatchResult(True, token, message)

    def complete_activation(self, token: str, password: str, display_name: Optional[str] = None) -> Tuple[bool, str]:
        payload = self._consume_token("activation", token.strip())
        if not payload:
            return False, "Activation token is invalid or has expired."
        username = str(payload["username"])
        record = self._find_user(username)
        if record is None:
            return False, "No matching account for activation token."
        salt, pw_hash = self._hash_password(password)
        record["salt"] = salt
        record["password_hash"] = pw_hash
        record["active"] = True
        if display_name:
            record["name"] = display_name
        self._save()
        return True, "Account activated successfully."

    def complete_activation_with_username(
        self, token: str, username: str, password: str, display_name: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Complete activation when the token was requested by email only."""
        payload = self._consume_token("activation", token.strip())
        if not payload:
            return False, "Activation token is invalid or has expired."
        email = str(payload.get("email", "")).strip()
        if not email:
            return False, "Activation token is missing email context."
        if self._username_taken(username):
            return False, "That username is already taken. Choose another."
        if not self._allowed_email(email):
            return False, "This email is not authorized for activation."
        record = self._find_user_by_email(email)
        salt, pw_hash = self._hash_password(password)
        if record is None:
            record = {
                "username": username,
                "email": email,
                "name": display_name or username,
                "salt": salt,
                "password_hash": pw_hash,
                "active": True,
            }
            self._users.append(record)
        else:
            record["username"] = username
            record["salt"] = salt
            record["password_hash"] = pw_hash
            record["active"] = True
            if display_name:
                record["name"] = display_name
        self._save()
        return True, "Account activated successfully."

    def request_password_reset(self, username: str, email: str, notifier: EmailNotifier) -> TokenDispatchResult:
        username = username.strip()
        email = email.strip()
        record = self._find_user(username)
        if record is None or not record.get("active"):
            return TokenDispatchResult(False, "", "No active account matches the provided credentials.")
        if str(record.get("email", "")).strip().lower() != email.lower():
            return TokenDispatchResult(False, "", "Email does not match our records.")

        token = self._issue_token(
            "reset",
            {
                "username": username,
                "email": email,
            },
        )
        email_body = (
            "You requested a password reset for the Deposit DCF analysis dashboard.\n\n"
            f"Reset token: {token}\n"
            "Enter this token in the application to choose a new password. The token expires in one hour."
        )
        sent = notifier.send_token(
            to_address=email,
            subject="Deposit DCF dashboard password reset",
            body=email_body,
        )
        if sent:
            message = "Password reset email sent."
        else:
            message = (
                "Password reset email failed to send. Verify the SMTP credentials and share this token: "
                f"`{token}`"
            )
        return TokenDispatchResult(sent, token, message)

    def complete_password_reset(self, token: str, password: str) -> Tuple[bool, str]:
        payload = self._consume_token("reset", token.strip())
        if not payload:
            return False, "Password reset token is invalid or has expired."
        username = str(payload["username"])
        record = self._find_user(username)
        if record is None:
            return False, "No matching account for password reset token."
        salt, pw_hash = self._hash_password(password)
        record["salt"] = salt
        record["password_hash"] = pw_hash
        record["active"] = True
        self._save()
        return True, "Password updated successfully."

    # ------------------------------------------------------------------ utilities
    def add_user(self, username: str, email: str, name: str, active: bool = False) -> Tuple[bool, str]:
        username = username.strip()
        email = email.strip()
        if self._find_user(username):
            return False, "Username already exists."
        self._users.append(
            {
                "username": username,
                "email": email,
                "name": name or username,
                "salt": "",
                "password_hash": "",
                "active": active,
            }
        )
        self._save()
        return True, "User added."

    def users(self) -> List[UserRecord]:
        return [
            UserRecord(
                username=str(item.get("username")),
                name=str(item.get("name") or item.get("username")),
                email=str(item.get("email") or ""),
                salt=str(item.get("salt") or ""),
                password_hash=str(item.get("password_hash") or ""),
                active=bool(item.get("active", False)),
            )
            for item in self._users
        ]

    def set_password_direct(self, username: str, password: str, display_name: Optional[str] = None, email: Optional[str] = None) -> Tuple[bool, str]:
        record = self._find_user(username)
        if record is None:
            return False, "User not found."
        salt, pw_hash = self._hash_password(password)
        record["salt"] = salt
        record["password_hash"] = pw_hash
        record["active"] = True
        if display_name:
            record["name"] = display_name
        if email:
            record["email"] = email
        self._save()
        return True, "Password updated successfully."

    def notifier(self) -> EmailNotifier:
        return EmailNotifier(self._data.get("smtp") or {})
