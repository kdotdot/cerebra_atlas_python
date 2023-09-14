import requests
import logging
import numpy as np


# https://stackoverflow.com/questions/38511444/python-download-files-from-google-drive-using-url
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def setup_logging(level=logging.DEBUG):
    logging.basicConfig(
        level=level,
        format=" [%(levelname)s] %(asctime)s.%(msecs)02d %(module)s - %(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def move_volume_from_LIA_to_RAS(volume, _affine=None):
    volume = np.rot90(volume, -1, axes=(1, 2))
    volume = np.flipud(volume)
    if _affine is None:
        return volume

    affine = _affine.copy()
    # Switch from LIA to RIA
    affine[0, -1] = 126  # Fix translation
    affine[0, 0] = 1

    # Switch from RIA to RSA
    affine[1, -1] = 256 - affine[2, -1]
    affine[2, 1] = 1

    # Switch from RSA to RAS
    aff = affine.copy()
    aff[1, :] = affine[2, :]
    aff[2, :] = affine[1, :]
    # affine[1, :], affine[2, :] = affine[2, :], affine[1, :] how?

    return volume, aff


if __name__ == "__main__":
    file_id = "13rfrvxVQe18ss2hccPy10DkKQdnNyjWL"
    destination = "./cer.mgz"
    download_file_from_google_drive(file_id, destination)
