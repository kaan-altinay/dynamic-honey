class BaseGenerator:
    def __init__(self):
        pass

    async def generate_bundle(self, host, path, site_profile):
        """
        Return generated bundle data for a meta-miss request.

        Expected return shape:
            GeneratedBundle(primary_path=..., artifacts=[...])
        """
        raise NotImplementedError("Generator model is not configured")