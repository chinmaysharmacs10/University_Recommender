from django import forms

class StudentForm(forms.Form):
    GRE_Verbal_Score = forms.IntegerField()
    GRE_Quantitative_Score = forms.IntegerField()
    GRE_Analytical_Writing_Score = forms.FloatField()
    CGPA = forms.FloatField()
    TOEFL_Score = forms.IntegerField()
    Research_Experience = forms.IntegerField(help_text="in months")
    Industry_Experience = forms.IntegerField(help_text="in months")
    Intern_Experience = forms.IntegerField(help_text="in months")

