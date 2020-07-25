import inspect

def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }

# Usage:
# get_default_args(LinearRegression.__init__)
# {'fit_intercept': True, 'normalize': False, 'copy_X': True, 'n_jobs': None}
