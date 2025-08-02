from ...interfaces import BaseBackendIdentifierMixin


class SqliteBackendIdentifierMixin(BaseBackendIdentifierMixin):
    backend_identifier: str = "sqlite"
