"""
Configuration variables shared across modules.

Contains global constants like model names, flags for data filtering, etc.
"""
EXCLUDE_NOISE_QUESTIONS=False #If True it filter out questions in which the disfluent text is '#Value' or 'question n/a'. Otherwise, use all questions
model_used="google/t5-efficient-tiny"  # The pre-trained model identifier used in the project which were : t5-small,t5-base and google/flan-t5-base