class BaseGenerator:
    def __init__(self):
        pass

    async def generate_page(self, host, path, site_profile):
        """
        Return generated page data for a meta-miss request.

        Expected return shape:
            {
                "path": "/requested/path",
                "headers": [{"Content-Type": "text/html"}],
                "body_bytes": b"..."
            }
        """
        raise NotImplementedError("Generator model is not configured")