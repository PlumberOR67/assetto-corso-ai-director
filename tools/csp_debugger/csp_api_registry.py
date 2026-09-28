"""Registry of the CSP API functions the Lua validator knows about.

Each entry maps ``namespace.function`` to ``{version, context, description,
required, returns}``. The validator reports any call that is not listed here as
unknown, so a function missing from this table is a warning, not an error.
"""

CSP_API_REGISTRY = {
    # ac.* namespace
    "ac.getPatchVersion": {
        "version": "0.1.0+",
        "context": "always",
        "description": "Get the CSP version",
        "required": True,
        "returns": "int",
    },
    "ac.getSim": {
        "version": "0.1.0+",
        "context": "always",
        "description": "Get the simulation object",
        "required": False,
        "returns": "SimObject",
    },
    "ac.getTrackSpline": {
        "version": "0.1.78+",
        "context": "replay,live",
        "description": "Get the track spline",
        "required": False,
        "returns": "TrackSpline",
    },
    "ac.getTrackName": {
        "version": "0.1.0+",
        "context": "always",
        "description": "Get the track name",
        "required": False,
        "returns": "string",
    },
    "ac.getLayoutName": {
        "version": "0.1.0+",
        "context": "always",
        "description": "Get the layout name",
        "required": False,
        "returns": "string",
    },
    "ac.getDocumentsPath": {
        "version": "0.1.0+",
        "context": "always",
        "description": "Get the path to Documents",
        "required": False,
        "returns": "string",
    },
    "ac.grabCamera": {
        "version": "0.1.78+",
        "context": "replay,live",
        "description": "Take control of the camera",
        "required": False,
        "returns": "CameraObject",
    },
    "ac.log": {
        "version": "0.1.0+",
        "context": "always",
        "description": "Write to the log",
        "required": False,
        "returns": "void",
    },
    # render.* namespace
    "render.createRay": {
        "version": "0.2.0+",
        "context": "replay,live",
        "description": "Create a ray for raycasting",
        "required": False,
        "returns": "RayObject",
    },
    # Ray methods
    "ray:physics": {
        "version": "0.2.0+",
        "context": "replay,live",
        "description": "Test the ray against physics",
        "required": False,
        "returns": "hitPos, hitNormal",
    },
    "ray:track": {
        "version": "0.2.0+",
        "context": "replay,live",
        "description": "Test the ray against the track",
        "required": False,
        "returns": "distance",
    },
    "ray:position": {
        "version": "0.2.0+",
        "context": "replay,live",
        "description": "Get the hit position",
        "required": False,
        "returns": "vec3",
    },
    "ray:normal": {
        "version": "0.2.0+",
        "context": "replay,live",
        "description": "Get the surface normal",
        "required": False,
        "returns": "vec3",
    },
    # ui.* namespace
    "ui.text": {
        "version": "0.1.0+",
        "context": "always",
        "description": "Draw text in the UI",
        "required": False,
        "returns": "void",
    },
    "ui.textWrapped": {
        "version": "0.1.0+",
        "context": "always",
        "description": "Draw wrapped text",
        "required": False,
        "returns": "void",
    },
    "ui.button": {
        "version": "0.1.0+",
        "context": "always",
        "description": "Draw a button",
        "required": False,
        "returns": "bool",
    },
    "ui.separator": {
        "version": "0.1.0+",
        "context": "always",
        "description": "UI separator",
        "required": False,
        "returns": "void",
    },
    # sim.* namespace (through ac.getSim())
    "sim.isReplayActive": {
        "version": "0.1.0+",
        "context": "always",
        "description": "Whether a replay is playing",
        "required": False,
        "returns": "bool",
    },
    "sim.timeSeconds": {
        "version": "0.1.0+",
        "context": "replay,live",
        "description": "Current session time",
        "required": False,
        "returns": "float",
    },
    # vec3 type
    "vec3": {
        "version": "0.1.0+",
        "context": "always",
        "description": "Vector type",
        "required": False,
        "returns": "vec3",
    },
    # setTimeout
    "setTimeout": {
        "version": "0.1.0+",
        "context": "always",
        "description": "Call a function after a delay",
        "required": False,
        "returns": "void",
    },
}

# Functions that need a particular context
CONTEXT_RESTRICTED = {
    "ac.getTrackSpline": ["replay", "live"],
    "ac.grabCamera": ["replay", "live"],
    "render.createRay": ["replay", "live"],
    "ray:physics": ["replay", "live"],
    "ray:track": ["replay", "live"],
}

# Functions that may be missing in older versions
VERSION_RESTRICTED = {
    "0.2.0+": [
        "render.createRay",
        "ray:physics",
        "ray:track",
        "ray:position",
        "ray:normal",
    ],
    "0.1.78+": [
        "ac.getTrackSpline",
        "ac.grabCamera",
    ],
}
