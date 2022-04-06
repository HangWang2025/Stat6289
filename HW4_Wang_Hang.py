#%%
# conda install pymagnitude
print('hello World')
# %%
from matplotlib.pyplot import magnitude_spectrum
from pymagnitude import *
# %%
file_path = 'GoogleNews-vectors-negative300.magnitude'
vectors = Magnitude(file_path)
# %%
len(vectors)

#%%
vectors.most_similar(positive = ["leg","throw"], negative = ["jump"])
#'forearm'


# [('forearm', 0.48294652),
#  ('shin', 0.47376168),
#  ('elbow', 0.4679689),
#  ('metacarpal_bone', 0.4678148),
#  ('metacarpal_bones', 0.46605825),
#  ('ankle', 0.46434423),
#  ('shoulder', 0.46183357),
#  ('thigh', 0.45393687),
#  ('knee', 0.44557068),
# #  ('ulna_bone', 0.44234914)]
# %%
vectors.doesnt_match( ['tissue', 'papyrus', 'manila', 'newsprint', 'parchment', 'gazette'])
#'tissue'
#%%
# vectors.most_similar_cosmul(positive = ["leg","jump"], negative = ["throw"])
#%%
vectors.dim
#300
#%%
vectors.most_similar("picnic", topn = 5)
#
# [('picnics', 0.7400874),
#  ('picnic_lunch', 0.721374),
#  ('Picnic', 0.7005339),
#  ('potluck_picnic', 0.6683274),
#  ('picnic_supper', 0.65189123)]
#%%