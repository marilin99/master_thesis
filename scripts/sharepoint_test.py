from office365.runtime.auth.authentication_context import AuthenticationContext
from office365.sharepoint.client_context import ClientContext
from office365.sharepoint.files.file import File

sharepoint_base_url = 'https://tartuulikool.sharepoint.com/sites/MarilinWork/'
sharepoint_user = 'marilinm@ut.ee'
sharepoint_password = 'Mokkatalu89'
folder_in_sharepoint = 'https://tartuulikool.sharepoint.com/:f:/r/sites/MarilinWork/Shared%20Documents/Marilin_training_set/15june2022?csf=1&web=1&e=8jixe9'


auth = AuthenticationContext(sharepoint_base_url) 
auth.acquire_token_for_user(sharepoint_user, sharepoint_password)
ctx = ClientContext(sharepoint_base_url, auth)
web = ctx.web
ctx.load(web)
ctx.execute_query()
print('Connected to SharePoint: ',web.properties['Title'])