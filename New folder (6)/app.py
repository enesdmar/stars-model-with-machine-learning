import matplotlib.pyplot as plt
from flask import Flask, render_template, request, send_file
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from IPython.display import HTML
app = Flask(__name__)
import joblib
df = pd.read_csv("Stars.csv")
df.head(20)
print(df.head(19))


stringcols=["Color"]
stringcols2=["Spectral_Class"]
le=LabelEncoder()
le1=LabelEncoder()
le2=LabelEncoder()
df[stringcols] = df[stringcols].apply(le.fit_transform)
df[stringcols2] = df[stringcols2].apply(le1.fit_transform)
y = le2.fit_transform(df["Type"])


x=df.drop(['Type'], axis=1)

regression_model = linear_model.LinearRegression()

regression_model.fit(X = pd.DataFrame(df["A_M"]), 
                    y = df["Type"])
"""
train_prediction = regression_model.predict(X = pd.DataFrame(df["A_M"]))

df.plot(kind="scatter", x="A_M", y="Type")
plt.plot(df["A_M"], train_prediction, color="red")
plt.show()
df.head()
regression_model.fit(X = pd.DataFrame(df["R"]), 
                    y = df["Type"])
train_prediction = regression_model.predict(X = pd.DataFrame(df["R"]))
df.plot(kind="scatter", x="R", y="Type")
plt.plot(df["R"], train_prediction, color="green")
plt.show()
regression_model.fit(X = pd.DataFrame(df["Color"]), 
                    y = df["Type"])
train_prediction = regression_model.predict(X = pd.DataFrame(df["Color"]))
df.plot(kind="scatter", x="Color", y="Type")

plt.show()
regression_model.fit(X = pd.DataFrame(df["Temperature"]), 
                    y = df["Type"])
train_prediction = regression_model.predict(X = pd.DataFrame(df["Temperature"]))
df.plot(kind="scatter", x="Temperature", y="Type")

plt.show()
regression_model.fit(X = pd.DataFrame(df["Spectral_Class"]), 
                    y = df["Type"])
train_prediction = regression_model.predict(X = pd.DataFrame(df["Spectral_Class"]))
df.plot(kind="scatter", x="Spectral_Class", y="Type")
plt.plot(df["Spectral_Class"], train_prediction, color="red")
plt.show()
regression_model.fit(X = pd.DataFrame(df["L"]), 
                    y = df["Type"])
train_prediction = regression_model.predict(X = pd.DataFrame(df["L"]))
df.plot(kind="scatter", x="L", y="Type")

plt.show()
"""
x_train,x_test,y_train,y_test = train_test_split(x, y, test_size=0.3, random_state=0)

tree=DecisionTreeClassifier(criterion='entropy')
classifier=tree.fit(x_train,y_train)

classifier.score(x_test,y_test)


print(list(classifier.predict(x_test)),end='')

print(list(y_test))

plt.figure(figsize=(12,8))

from sklearn import tree

tree.plot_tree(classifier.fit(x_train, y_train)) 

y_pred_en = classifier.predict(x_test)



@app.route('/')
def my_form(): 
    return render_template('TEZ_HTML.html', ut="Result",img="",desc="Star Classification")

@app.route('/', methods=['POST'])
def my_form_post():
    
        temp = int(request.form['varT'].strip())
        ls = float(request.form['varL'].strip())
        rr = float(request.form['varR'].strip())
        abm = float(request.form['varAM'].strip())
        a5=request.form['varC'].strip()
        a6=request.form['varS'].strip()
        sg = le.transform([request.form['varC'].strip()])[0]
        ms = le1.transform([request.form['varS'].strip()])[0]
        startype=classifier.predict([[temp,ls,rr,abm,sg,ms]])[0]
        result=""
        image=""
        desc=""
        if startype==0:
            result="Red Dwarf"
            image="red dwarf.jpg"
            desc="""Red Dwarfs: The Most Common Stars in the Universe
Red dwarfs are the smallest and coolest stars in the main sequence. They are much smaller than the Sun, with a radius only about 15% of the Sun's radius. Due to their low mass, they burn hydrogen very slowly, which means they can live for trillions of years. In fact, they are the most common type of star in the universe, estimated to make up about 75% of our galaxy."""
        elif startype==1:
            result="Brown Dwarf"
            image="brown dwarf.jpg"
            desc="""Contribution of Brown Dwarfs to Evolution:

Provide Insights into Star Formation: Brown dwarfs are frequently found within star-forming clouds. Therefore, they serve as an important source for gaining insights into the processes of star formation.

Provide Information about Mass Distribution of Stars: The discovery of brown dwarfs has assisted astronomers in developing more accurate models regarding the mass distribution of stars.

Provide Insights into Planet Formation: Some brown dwarfs have been found to host planets in their orbits. These findings aid in our understanding of the processes involved in planet formation."""
        elif startype==2:
            result="White Dwarf"
            image="white dwarf.jpg"
            desc="""White Dwarfs: The End of Cooling Stars
White dwarfs are cold and dense stars in the final stage of their lives. Sun-like stars enter the red giant stage after they have exhausted their nuclear fuels. During the red giant stage, the star expands greatly, and as it collapses inward to become a white dwarf, the compression of helium around the star's core increases until it reaches a point of explosion, after which it sheds its outer layers into space, leaving behind the remaining core, which becomes a white dwarf."""
        elif startype==3:
            result="Main Sequence"
            image="main sequence.jpg"
            desc="""Main Sequence: The Stable Life of Stars
The main sequence is the longest and most stable stage of a star's life cycle. During this stage, stars undergo nuclear fusion of hydrogen in their cores. This fusion reaction powers the star, causing it to emit light and heat.

The most common types of stars in the universe, including our Sun, are found on the main sequence. Stars remain on the main sequence for different durations depending on their masses. Larger mass stars spend shorter periods on the main sequence, while smaller mass stars can remain on the main sequence for trillions of years."""
        elif startype==4:
            result="Super Giants"
            image="super giant.jpg"
            desc="""Super Giants: The Magnificent Beasts of the Cosmos
Super giants are the largest and brightest stars in the universe. They can be millions of times larger than the Sun and produce incredibly high amounts of energy. These stars are in the final stages of their lives, with their nuclear fuel in the core nearing depletion."""
        elif startype==5:
            result="Hyper Giants"
            image="hyper giant.jpg"
            desc="""Hyper Giants: The Magnificent Titans of the Universe
Hyper giants are the largest and rarest type of stars in the universe. They can be hundreds of millions of times larger than the Sun and produce incredibly high amounts of energy. These stars are in the final stages of their lives, with their nuclear fuel in the core nearing depletion."""
            
        #result = df.to_html().replace("<table ",'<table class="table table-bordered" ').replace("<thead>",'<thead class="thead-dark">')
        return render_template('TEZ_HTML.html', ut=result,img=image,desc=desc,a1=temp,a2=ls,a3=rr,a4=abm,a5=a5,a6=a6)
    
            
            
          
        


if __name__ == '__main__':
    app.run(debug=True)