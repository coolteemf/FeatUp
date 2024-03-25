from setuptools import setup, Extension, Command
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension

# ext_modules=[
    # CUDAExtension(
    #     'featup.adaptive_conv_cuda.cuda_impl',
    #     [
    #         'featup/adaptive_conv_cuda/adaptive_conv_cuda.cpp',
    #         'featup/adaptive_conv_cuda/adaptive_conv_kernel.cu',
    #     ]),
    # CppExtension(
    #     'featup.adaptive_conv_cuda.cpp_impl',
    #     ['featup/adaptive_conv_cuda/adaptive_conv.cpp'],
    #     undef_macros=["NDEBUG"]),
# ],
# print(type(ext_modules[0]))
# cmdclass={
#     'build_ext': BuildExtension
# }

setup(
    name='featup',
    version='0.1',
    packages=['featup'],
    ext_modules=[CppExtension(
        'featup.adaptive_conv_cuda.cpp_impl',
        ['featup/adaptive_conv_cuda/adaptive_conv.cpp'],
        undef_macros=["NDEBUG"]),],
    cmdclass={'build_ext': BuildExtension}
)

# def build(setup_kwargs):
#     """
#     This function is mandatory in order to build the extensions.
#     """
#     setup_kwargs.update(
#         {"ext_modules": ext_modules, "cmdclass": cmdclass}
#     )