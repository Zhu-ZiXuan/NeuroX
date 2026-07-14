"""Reserved config keys.

The three magic keys the serialization subsystem reserves in config mappings:
the polymorphic class discriminator and the two fragment-composition directives.

See also:
    docs/internals/common/serialize/README.md
"""

CLASS_DISCRIMINATOR = "_neurox_class"
USE_DIRECTIVE = "_neurox_use"
USE_PRESET_DIRECTIVE = "_neurox_use_preset"
