from django.shortcuts import render
from django.http import JsonResponse
import pandas as pd
from .models import PredResults
import os


def predict(request):
    return render(request, 'predict.html')


def predict_chances(request):
    if request.POST.get('action') == 'post':
        # Receives the input from the form
        sepal_length = float(request.POST.get('sepal_length'))
        sepal_width = float(request.POST.get('sepal_width'))
        petal_length = float(request.POST.get('petal_length'))
        petal_width = float(request.POST.get('petal_width'))

        pickle_file_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'new_model.pickle')
        # Unpickle the model
        model = pd.read_pickle(pickle_file_path)

        # Make prediction
        result = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])

        classification = result[0]

        PredResults.objects.create(sepal_length=sepal_length, sepal_width=sepal_width, petal_length=petal_length,
                                   petal_width=petal_width, classification=classification)

        return JsonResponse({'result': classification, 'sepal_length': sepal_length, 'sepal_width': sepal_width,
                             'petal_length': petal_length, 'petal_width': petal_width}, safe=False)
    else:
        return JsonResponse({'error': 'Invalid request'}, status=400)


def view_results(request):
    data = {"dataset": PredResults.objects.all()}
    return render(request, "results.html", data)
