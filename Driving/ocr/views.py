# -*- coding: utf-8 -*-
from django.template import RequestContext
from django.http import HttpResponseRedirect,HttpResponse
from django.urls import reverse
from django.shortcuts import render
from django.core import serializers
from ocr.models import Document
from ocr.forms import DocumentForm
from django.core import serializers
import json
from ocr.prediction import get_data

import base64


def list(request):
    # Handle file upload
    if request.method == 'POST':
        form = DocumentForm(request.POST, request.FILES)
        if form.is_valid():
            newdoc = Document(docfile = request.FILES['docfile'])
            print('sfsefwe-->',request.FILES['docfile'])
            newdoc.save()
            result_det=get_data('Driving/media/documents/'+str(request.FILES['docfile']))
            result_det.append(str(request.FILES['docfile']))
            # with open("/home/hitesh/Desktop/Driving/Driving/media/documents/face.jpg", "rb") as image_file:
            #     encoded_string = base64.b64encode(image_file.read())

            """'image':str(encoded_string),""" #add to below if you want
            dict_result={'name':result_det[0],'license_num':result_det[1],'DOB&Gen':result_det[2],'Date_issue':result_det[5]}
            dict_result=json.dumps(dict_result)
            return render(
                request,
                'ocr/result.html',
                { 'res': result_det,'json':dict_result }
            )
    else:
        form = DocumentForm() # A empty, unbound form

    # Load documents for the list page
    documents = Document.objects.all()
    return render(
        request,
        'ocr/list.html',
        {'documents': documents, 'form': form}
    )
    # return HttpResponse(data)

def index(request):
    return HttpResponseRedirect('ocr/index.html')
