from flask import Flask, render_template, redirect, request, session, url_for
from flask_bootstrap import Bootstrap
from flask_dropzone import Dropzone
from flask_uploads import UploadSet, configure_uploads, IMAGES, patch_request_class
import os

import config
import get_img


app = Flask(__name__)
dropzone = Dropzone(app)
bootstrap = Bootstrap(app)

# Dropzone settings
app.config['DROPZONE_UPLOAD_MULTIPLE'] = True
app.config['DROPZONE_ALLOWED_FILE_CUSTOM'] = True
app.config['DROPZONE_ALLOWED_FILE_TYPE'] = 'image/*'
app.config['DROPZONE_REDIRECT_VIEW'] = 'index'
# Uploads settings
app.config['UPLOADED_PHOTOS_DEST'] = os.getcwd() + '/uploads'
app.config['SECRET_KEY'] = config.super_key

photos = UploadSet('photos', IMAGES)
configure_uploads(app, photos)
patch_request_class(app)

def run_model():
    trans_urls, refl_urls = get_img.save_imgs(session['file_urls'])

    # session.pop('file_urls', None)

    session['trans_urls'] = trans_urls
    session['refl_urls'] = refl_urls


@app.route('/', methods=['GET', 'POST'])
def index():

    if 'file_urls' not in session:
        session['file_urls'] = []
    if 'trans_urls' not in session:
        session['trans_urls'] = []
    if 'refl_urls' not in session:
        session['refl_urls']=  []
    if 'for_rem' not in session:
        session['for_rem'] = []

    for link in session['for_rem']:
        os.remove(f'./uploads/{link[1]}')
    session['for_rem'] = []

    file_urls = session['file_urls']

    if request.method == 'POST':
        file_obj = request.files
        for f in file_obj:
            file = request.files.get(f)
            filename = photos.save(
                file,
                name=file.filename
            )

            file_urls.append((photos.url(filename), file.filename))

        session['file_urls'] = file_urls
        run_model()
        
        return 'uploading...'
        

    if 'trans_urls' in session and session['trans_urls'] != []:
        print('KEEEEEEEK')
        t_urls = session['trans_urls']
        r_urls = session['refl_urls']
        f_urls = session['file_urls']

        session['for_rem'] = session['file_urls'] + session['trans_urls'] + session['refl_urls']

        session.pop('file_urls')
        session.pop('trans_urls')
        session.pop('refl_urls')

        return render_template('index.html', file_urls=f_urls, trans_urls=t_urls, refl_urls=r_urls, is_post=True)
    
    return render_template('index.html')

@app.route('/reset')
def reset():
    session['for_rem'] = []
    session['file_urls'] = []
    session['trans_urls'] = []
    session['refl_urls'] = []
    return 'reset is done'



if __name__ == '__main__':
    app.run(debug=True)
