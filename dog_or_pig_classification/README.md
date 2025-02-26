# Is it a dog or pig?

First, I set up the training data for a simple classification problem, where I tried to distinguish between pigs and dogs using three features: long fur, short legs, and whether it makes a "woof" sound. The features are binary, represented by 1 (yes) or 0 (no).

```python
porco1 = [0, 1, 0]
porco2 = [0, 1, 1]
porco3 = [1, 1, 0]

cachorro1 = [0, 1, 1]
cachorro2 = [1, 0, 1]
cachorro3 = [1, 1, 1]

# 1 -> porco, 0 -> cachorro
dados = [porco1, porco2, porco3, cachorro1, cachorro2, cachorro3]
classes = [1, 1, 1, 0, 0, 0]
```

I used the LinearSVC model from scikit-learn to train with the provided data. After training, I made the prediction for an animal with unknown characteristics (all values ​​are 0).

```python
from sklearn.svm import LinearSVC

modelo = LinearSVC()
modelo.fit(dados, classes)

animal_misterioso = [0, 0, 0]
modelo.predict([animal_misterioso])
```

Next, I define three new animals with different characteristics and use the trained model to make predictions about these animals. I will compare these predictions with the real classes to evaluate the model's performance.

```python
misterio1 = [1, 1, 1]
misterio2 = [1, 1, 0]
misterio3 = [0, 1, 1]

testes = [misterio1, misterio2, misterio3]
previsoes = modelo.predict(testes)

testes_classes = [0, 1, 1]
```

I calculated the accuracy manually by comparing the predictions with the actual classes. Accuracy is the proportion of correct predictions to the total predictions.

```python
corretos = (previsoes == testes_classes).sum()
total = len(testes)
taxa_de_acerto = corretos / total * 100

print(f"Acurácia: {taxa_de_acerto:.2f}%")
```

Finally, I used scikit-learn's accuracy_score function to calculate accuracy in a more straightforward and robust way, which confirms our manual calculations.

```python
from sklearn.metrics import accuracy_score

taxa_de_acerto = accuracy_score(testes_classes, previsoes) * 100
print(f"Acurácia: {taxa_de_acerto:.2f}%")
```

## Conclusion

With this small project I learned to:

- Conceptualize what supervised learning and binary classification are;
- Use item characteristics to define classes and estimate the classes of new items using estimators;
- Apply the scikit-learn library and the LinearSVC class to create an estimator;
- Calculate the accuracy of the model using the accuracy_score function.
