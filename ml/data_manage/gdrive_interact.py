# AUTOGENERATED! DO NOT EDIT! File to edit: util_nbs/99b_data_manage.gdrive_interact.ipynb (unless otherwise specified).

__all__ = ['gauth', 'cred_fpath', 'drive', 'gtypes', 'get_root_remote_id', 'get_folder_id', 'grab_folder_contents',
           'check_file_exists_remote', 'get_file_id', 'download_file', 'upload_new_file', 'update_existing_file',
           'sync_file_to_remote']

# Cell
from pydrive.drive import GoogleDrive
from pydrive.auth import GoogleAuth
import os

from ..local_repo_path import local_repo_path

# Cell
gauth = GoogleAuth()
# this needs to be added to the root of the repo
cred_fpath = local_repo_path + 'client_secrets.json'
# tell pydrive where to look for it
gauth.DEFAULT_SETTINGS['client_config_file'] = cred_fpath
# initiate the drive object and open the connection
drive = GoogleDrive(gauth)
gauth.LocalWebserverAuth() # Creates local webserver and auto handles authentication.

# Cell
gtypes = {
    'folder' : 'application/vnd.google-apps.folder'
}

# Cell
def get_root_remote_id(folderName = 'ml_repo_data', gtypes=gtypes):
    # query google drive
    folders = drive.ListFile(
        {'q': f"title='{folderName}' and mimeType='{gtypes['folder']}' and trashed=false"}).GetList()
    folder = folders[0] # the above returns a list
    return folder['id']

# Cell
def get_folder_id(parent_id, foldername):
    # grab the folder
    ftype = gtypes['folder'] # unfortunately if I don't do this Jupyter freaks out with indentations/coloration
    folders = drive.ListFile(
        {'q': f"title='{foldername}' and mimeType='{ftype}' and '{parent_id}' in parents and trashed=false"}).GetList()
    folder = folders[0] # the above returns a list
    return folder['id']

# Cell
def grab_folder_contents(parent_id):
    '''Return a list of all the items in a folder based on its parent id'''
    file_list = drive.ListFile({'q': f"'{parent_id}' in parents and trashed=false"}).GetList()
    return file_list

# Cell
def check_file_exists_remote(parent_id, fname):
    file_list = grab_folder_contents(parent_id)
    for file in file_list:
        if file['title'] == fname : return True
        continue
    return False

# Cell
def get_file_id(parent_id, fname):
    # grab the folder
    ftype = gtypes['folder'] # unfortunately if I don't do this Jupyter freaks out with indentations/coloration
    file_list = drive.ListFile(
        {'q': f"title='{fname}' and '{parent_id}' in parents and trashed=false"}).GetList()
    file = file_list[0] # the above returns a list
    return file['id']

# Cell
def download_file(file_id, local_dpath = None):
    # Create GoogleDriveFile instance with file id of file1.
    file = drive.CreateFile({'id': item['id']})
    local_dpath = './' if local_dpath is None else local_repo_path + local_dpath
    local_fpath = local_dpath + file['title']
    file.GetContentFile(local_fpath)
    return local_fpath

# Cell
def upload_new_file(local_fpath, fname, parent_id):
    file = drive.CreateFile({'parents': [{'id': f'{parent_id}'}]})
    file['title'] = fname
    file.SetContentFile(local_fpath)
    file.Upload()
    return

# Cell
def update_existing_file(local_fpath, file_id):
    file = drive.CreateFile({'id': item['id']})
    file.SetContentFile(local_fpath)
    file.Upload()
    return

# Cell
def sync_file_to_remote(local_fpath, fname, parent_id):
    '''will check if file exists remote first then will upload/update
    accordingly'''
    file_exists_remote = check_file_exists_remote(parent_id, fname)
    # update if its already there
    if file_exists_remote:
        file_id = get_file_id(parent_id, fname)
        update_existing_file(local_fpath, file_id)
    # upload a new one else
    else:
        upload_new_file(local_fpath, fname, parent_id)
    return