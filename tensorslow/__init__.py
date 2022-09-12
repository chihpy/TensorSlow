
is_simple_core = True
if is_simple_core:
    from tensorslow.core_simple import Variable
    from tensorslow.core_simple import Function
    from tensorslow.core_simple import using_config
    from tensorslow.core_simple import no_grad
    from tensorslow.core_simple import as_array
    from tensorslow.core_simple import as_variable
    from tensorslow.core_simple import setup_variable
else:
    print("unimplement")

setup_variable()