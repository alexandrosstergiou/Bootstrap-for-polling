# Bootstrap-for-polling

One of the most straightforward ways of determining the voter's intentions has been the use polls. Ambivalent voters
and voters that have not yet thought of their decision can play a part in the misinterpretation of the data obtained.
Features of these examples in the data such as their previous voting’s, their beliefs or their education can have a 
major impact on the tend that people can base their own opinion. This work is centred towards using these information
to show that a correlation does indeed exist between individual personal traits and voting intentions. Furthermore, 
with the aid of data science techniques such as bootstrapping to normalise the sample to resemble the population's
features an accurate estimate can be acquired.

To achieve these results two methods were tested:

  - Bootstrapping new results based on dataset elements in order to create new random elements.
  For each feature category threshold values are used.
  - Bootstrapping with Nearest Neighbour approach for predicting the most probable voting result in newly generated example.
  This approach increases the variance in the data compared to the random Bootstrapping technique. Threshold values are still used to
  control the number of new examples per feature value.

### The datasets used

* [2015 General Election results] - This dataset includes the previous election results and each party’s percentages.
* [Education qualification demographics] - These are grouped to six distinct qualification levels based on the FHEQ framework. Among the percentages for each education level, the number of people is also displayed as well as the rank of each ministry’s rank per level group.
* [Gender and age ratios] - The data in this file include the number of people and the gender ratio of each age.
* [Social grades indexes] - These are the indicators provided by the NRS that show the income and the financial state for a group of people.
* [Newspaper preference] - Survey estimates for the most popular newspapers that are being read in the UK.
* [Station ratings based on fondness] - These are the ratings of the most favourable TV news stations in the United kingdom based on 2008 BARB survey.

|  DATASET |NUMBER OF FEATURES|USED FEATURES|
|:---------|:----------------:|:-----------:|
|  YouGov  |        30        |      7      |
| 2015 Vote|         2        |      1      |
|FHEQ Level|        22        |      6      |
|Gender/age|         2        |      2      |
|Soc. Grade|         5        |      4      |
|Newspapers|         3        |      1      |
|  TV news |         1        |      1      |

Consequently the characteristics that are bootstapped are:

  - The age of each person in the dataset.
  - The gender of the voter
  - What individual education qualifications they
have obtained, taking into account university degrees, diplomas, certificates, apprenticeships, etc.
  - The social grade group that each person is assigned.
  - The vote casted in the 2015 general election.
  - What news station they prefer to watch.
  - The newspaper that they buy or read more
frequently.


### Methodology

In each stage a condition is used to find if the data is representative of the general population. So the condition can take three different values:

  - **The features are found to be over-represented.** In this case nothing can be done to normalise the data as it is not desired to delete examples, since information will be lost in that case. Instead, the method will continue to the next characteristic, because if some features are found to be over-represented, consequently others will be found to be under- represented.
  - **Attributes are found to be equal between the two sets.** This is the ideal situation that we desire. Once reached at this case, a flag is raised which will be validated in the next iteration to find if a satisfactory number of conditions are true for the new generated data.
  - **The characteristics are under-represented**. The final case is that the survey data do not hold adequate data for a feature and therefore, it is required to bootstrap new examples. Some attributes however need to be treated with care, for example to reassure that a person that did not voted does not include a voting party or that an age falls within the age group and is not simply the top or bottom value of the group.

![Alt text](/path/to/img.jpg)

### Dependencies, Code and Usage

The main dependencies are:

```python
import numpy as np
import pandas as pd
import random
```

Loading datasets with pandas:

```python
df_online = pd.read_csv("./NatRep_Online_Upload.csv", delimiter = ",")
df_phone = pd.read_csv("./NatRep_Phone_upload.csv", delimiter = ",")
```
```python
df_elections = pd.read_csv("./2015_general_elections/2015_voting_gen_election.csv", delimiter=",")
```
```python
df_qualifications = pd.read_csv("./Education_qualifications/UK_Qualifications.csv", delimiter=",")
```
```python
df_sex_to_age = pd.read_csv("./Gender_demographics_by_age/UK_M_to_F_ratio_by_age.csv", delimiter=",")
```
```python
df_newspaper = pd.read_csv("./Newspaper_readability/Newspaper_readerships_uk.csv", delimiter=",")
```
```python
df_social_grade = pd.read_csv("./Social_grade/Aproximated_social_grade.csv", delimiter=",")
```
```python
df_station_ratings = pd.read_csv("./Station_ratings/Station_Ratings_UK.csv", delimiter=",")
```

**Core methods used**:
```python
def find_precentages_in_data(data):
```
The method takes as an argument the polling data used as the base of the experiments and returns the percentages per category feature.
This is done in order to acquire a mean for evaluation through every iteration. Based on the percentages obtained in here,  the examples that have been over-represented or under-represented can be found through an analysis. A call by the main program should be made to this method at the end of each iteration.


```python
def bootstrap(data, sample_of_interest, condition):
```
The parent method for every feature that can be bootstrap. It requires the polling data to be used, the sample or the characteristic that will be bootstrapped and the condition that specifies which variable is of interest. For example the condition will be "== 35" if a new example is to be created of a person with an age of 35.

```python
def conditions(online_precentages, df_sex_to_agegroup, df_social_grade, df_elections, df_mean_qualifications, df_percent_newspapers, df_station_ratings):
```
This method is used as a flag  to find if the data generated are equivaent to the population statistics. Therefore, apart from the examples generated, it also requires the population data. It returns an array of conditions for each variable in the features questioned.


```python
def nn(data, example):
```
A very basic Nearest Neighbour method that finds the most similar voting intention class in the data based on the new generated example. Returns the new example with the most fitting class found assigned to it.

**The code is written with the aid of a Jupiter notebook and therefore Anaconda needs to be installed in the system as well as
the previously mentioned libraries. for more information visit:** (http://jupyter.org/)

License
----

MIT




[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)



   [2015 General Election results]: <http://www.bbc.co.uk/news/election/2015/results>
   [Education qualification demographics]: <http://www.nationalarchives.gov.uk/webarchive/>
   [Gender and age ratios]: <https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/bulletins/2011censuspopulationestimatesfortheunitedkingdom/2012-12-17>
   [Social grades indexes]: <http://www.nrs.co.uk/nrs-print/lifestyle-and-classification-data/social-grade>
   [Newspaper preference]: <http://www.newsworks.org.uk/Market-Overview>
   [Station ratings based on fondness]:<http://www.barb.co.uk/viewing-data/>

   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>

