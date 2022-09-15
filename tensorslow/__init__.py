
is_simple_core = False
if is_simple_core:
    from tensorslow.core_simple import Variable
    from tensorslow.core_simple import Function
    from tensorslow.core_simple import using_config
    from tensorslow.core_simple import no_grad
    from tensorslow.core_simple import as_array
    from tensorslow.core_simple import as_variable
    from tensorslow.core_simple import setup_variable
else:
    from tensorslow.core import Variable
    from tensorslow.core import Function
    from tensorslow.core import using_config
    from tensorslow.core import no_grad
    from tensorslow.core import as_array
    from tensorslow.core import as_variable
    from tensorslow.core import setup_variable

setup_variable()