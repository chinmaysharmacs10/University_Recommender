from django.shortcuts import render
import pandas as pd
import numpy as np
import joblib

from .forms import StudentForm

# Load the pickled classification model
classification_model = joblib.load('./models/classifier_model.pkl')

# Make the index view with the form
def index(request):
    form = StudentForm()

    context = {'univ':'', 'form':form}
    return render(request, 'index.html', context)


# When submit button is clicked go to predict_uni view
def predict_uni(request):
    if request.method == 'POST':
        form = StudentForm(request.POST)

        if form.is_valid():
            # store the data entered in form as a dictionary
            student = {}
            student['researchExp'] = form.cleaned_data.get('Research_Experience')
            student['industryExp'] = form.cleaned_data.get('Industry_Experience')
            student['toeflScore'] = form.cleaned_data.get('TOEFL_Score')
            student['internExp'] = form.cleaned_data.get('Intern_Experience')
            student['greV'] = form.cleaned_data.get('GRE_Verbal_Score')
            student['greQ'] = form.cleaned_data.get('GRE_Quantitative_Score')
            student['greA'] = form.cleaned_data.get('GRE_Analytical_Writing_Score')
            student['cgpa_4'] = form.cleaned_data.get('CGPA')

    test_data = pd.DataFrame({'x':student}).transpose()
    univ_index = classification_model.predict_classes(test_data)   # This will give us the label index of the University

    # Use the univ dictionary taken from (university_dict.ipynb) to get the name of the University based on label index
    univ = {0: 'Virginia Polytechnic Institute and State University',
            1: 'University of Wisconsin Madison',
            2: 'University of Washington',
            3: 'University of Utah',
            4: 'University of Texas Dallas',
            5: 'University of Texas Arlington',
            6: 'University of Southern California',
            7: 'University of Pennsylvania',
            8: 'University of North Carolina Charlotte',
            9: 'University of Minnesota Twin Cities',
            10: 'University of Massachusetts Amherst',
            11: 'University of Maryland College Park',
            12: 'University of Illinois Urbana-Champaign',
            13: 'University of Illinois Chicago',
            14: 'University of Florida',
            15: 'University of Colorado Boulder',
            16: 'University of Cincinnati',
            17: 'University of California Irvine',
            18: 'University of Arizona',
            19: 'Texas A and M University College Station',
            20: 'Syracuse University',
            21: 'SUNY Stony Brook',
            22: 'SUNY Buffalo',
            23: 'Rutgers University New Brunswick/Piscataway',
            24: 'Purdue University',
            25: 'Ohio State University Columbus',
            26: 'Northeastern University',
            27: 'North Carolina State University',
            28: 'New York University',
            29: 'New Jersey Institute of Technology',
            30: 'Georgia Institute of Technology',
            31: 'George Mason University',
            32: 'Cornell University',
            33: 'Clemson University',
            34: 'Carnegie Mellon University',
            35: 'Arizona State University'}

    univ_name = univ[univ_index[0]]      # This is the Name of the University

    context = {'univ': univ_name, 'form': form}
    return render(request,'index.html',context)
