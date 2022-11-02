import io
import pickle


class RenameUnpickler(pickle.Unpickler):
    """
    Obtained from: https://stackoverflow.com/questions/2121874/python-pickling-after-changing-a-modules-directory
    """
    def find_class(self, module, name):
        renamed_module = module
        if "torch_utils" in module:
            renamed_module = f"NVIDIA_STYLEGAN3.{module}"

        return super(RenameUnpickler, self).find_class(renamed_module, name)


def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()


def renamed_loads(pickled_bytes):
    file_obj = io.BytesIO(pickled_bytes)
    return renamed_load(file_obj)