from django import forms

class StudentForm(forms.Form):
    GRE_Verbal_Score = forms.IntegerField(help_text="out of 170")
    GRE_Quantitative_Score = forms.IntegerField(help_text="out of 170")
    GRE_Analytical_Writing_Score = forms.FloatField(help_text="out of 6.0")
    CGPA = forms.FloatField(help_text="in 4-scale")
    TOEFL_Score = forms.IntegerField(help_text="out of 120")
    Research_Experience = forms.IntegerField(help_text="in months")
    Industry_Experience = forms.IntegerField(help_text="in months")
    Intern_Experience = forms.IntegerField(help_text="in months")

