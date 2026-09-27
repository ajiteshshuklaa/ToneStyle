from django.shortcuts import render
from .predict import analyze_image
from django.core.files.storage import FileSystemStorage
import base64
import os
from django.conf import settings

def index(request):
    result = None

    if request.method == "POST":
        # Check if the form submitted webcam Base64 data
        if request.POST.get('webcam_image'):
            data_url = request.POST.get('webcam_image')
            header, imgstr = data_url.split(';base64,')
            
            filepath = "temp_webcam_capture.png"
            with open(filepath, "wb") as f:
                f.write(base64.b64decode(imgstr))
            
            tone, undertone, colors, scores = analyze_image(filepath)
            
            # Immediately delete the temporarily saved photo
            if os.path.exists(filepath):
                os.remove(filepath)
            
            result = {
                "tone": tone,
                "undertone": undertone,
                "colors": ", ".join(colors),
                "scores": scores
            }
            
        # Check if the form submitted a traditional file upload
        elif request.FILES.get('image'):
            image = request.FILES['image']
            fs = FileSystemStorage()
            filename = fs.save(image.name, image)
            filepath = fs.path(filename)

            tone, undertone, colors, scores = analyze_image(filepath)
            
            # Immediately delete the temporarily saved photo
            if os.path.exists(filepath):
                os.remove(filepath)

            result = {
                "tone": tone,
                "undertone": undertone,
                "colors": ", ".join(colors),
                "scores": scores
            }

    return render(request, "index.html", {"result": result})