
# Normalized saliency explanations 
> How the magnitude of scores assigned by saliency methods for XAI to tokens and features affects explanation consumption by end users.

Many papers in the field of eXplainable AI try to address the problem of identifying what's the best explanation for an end user.
For this sake, for example, [Kim et al.](https://arxiv.org/pdf/1702.08608.pdf) suggests to carry on different kinds of evaluation depending on the expertise of the target audience with respect to the system to be explained.
Others, like [Arya et al.](https://arxiv.org/pdf/1909.03012.pdf), suggest a more informed decision process that guides the right explanation to the right user for the appropriate use case.

In many cases however _saliency_ explanations are commonly the first medium to bridge the inner workings of a black box system and an end user.
Saliency explanations provide a score, aka saliency or feature attribution, to each feature in the input that was submitted to the black box system.
Well known saliency explanation systems are [LIME](https://www.kdd.org/kdd2016/papers/files/rfp0573-ribeiroA.pdf) and [SHAP](https://arxiv.org/abs/1705.07874).

Let's consider a text classifier which detects whether if a text should be considered spam or not.
In our example the classifier ingests a document divided into sentences and tokens, its output is _true_ if the text is considered spam, _false_ otherwise.
Going forward we use the [TrustyAI Explainability Toolkit](https://arxiv.org/abs/2104.12717) to generate predictions and explanations.

A suspicious text document needs to be analyzed by the spam classifier.
```java
List<Feature> features = new LinkedList<>();
Function<String, List<String>> tokenizer = s -> Arrays.asList(s.split(" "));
features.add(FeatureFactory.newFulltextFeature("s1", "greetings from Tatooine", tokenizer));
features.add(FeatureFactory.newFulltextFeature("s2", "we reach out from the outer hem to announce you jave won the rebel prize", tokenizer));
features.add(FeatureFactory.newFulltextFeature("s3", "you will get empire credits for free", tokenizer));
features.add(FeatureFactory.newFulltextFeature("s4", "you just need to send us 100 $ for the hyperspace flight", tokenizer));
PredictionInput input = new PredictionInput(features);
```

We pass the document divided into sentences to the classifier and get the resulting prediction.

```
PredictionProvider model = TestUtils.getDummyTextClassifier();
List<PredictionOutput> outputs = model.predictAsync(List.of(input)).get();
Prediction prediction = new SimplePrediction(input, outputs.get(0));        
```

The text is predicted as spam. We are now interested in understanding why it is the case.
For this reason we use the _LimeExplainer_ that generates a _Saliency_ object that holds the saliency score for each sentence and word contained in the input text.

```java
LimeExplainer limeExplainer = new LimeExplainer();
Map<String, Saliency> saliencyMap = limeExplainer.explainAsync(prediction, model).get();
```

Now if we look at the saliency scores we'll get the following:

| greetings | from | Tatooine | we | reach | out | from | the | outer | hem | to | announce | you've | just | won | the | rebel | prize | you | will | get | empire | credits | for | free | you | just | need | to | send | us | 100 | $ | for | the | hyperspace | flight |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.0 | 0.07486544544602479 | -0.07486544544602479 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | -0.0076090239036217455 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | -0.14158923531517434 | -0.008141655576875267 | 0.0 | 0.0 | 0.0 | 0.0 | 0.07486544544602479 | 0.07486544544602479 | 0.0 | 0.0 | 0.0 | 0.0 | 0.23220536024169616 | -0.0076090239036217455 | 0.08247446934964653 | 0.1497308908920496 | 0.0 | 


As we can see, each word has been assigned a _saliency score_ which indicates how important it is for that specific prediction.
For example, the most important token is the _'$'_ token, followed by _`hyperspace`_. These are the words that are most _responsible_ for the text being classified as spam.
However there is more information in the explanation: there are tokens which don't have any influence on the result (saliency score = 0), also there are some words having a negative importance for the prediction.
Zero salient words are unimportant, that's straightforward.
How can we interpret those negative scores ? 

In LIME as well as in SHAP those features having a negative score play an _adversarial role_, that is that they play "against" the final prediction. 
For example the word _Tatooine_ has a negative score, you can interpret like this: had the word _Tatooine_ not been present in the text, the classifier would have been even more convinced that this text is spam.

In addition to that, how can a user interpret the magnitudes of saliency scores? E.g., is _0.23_ representetive of high importance ?
For this aspect, it very much depends on the explainability technique used. 
LIME saliency scores are the weights of a linear classifier trained on the neighborhood of the input data, so those weights' magnitude would vary a lot depending on the specific input at hand.

Concepts like the one of adversarial features are not very comfortable for non technical users, also varying scores' magnitudes might confuse users.
To address these issues, one simple technique is to normalize the saliency scores, either in interval _\[-1, 1\]_ or _\[0, 1\]_.
Note that we won't normalize zero-scores (an unimportant feature has to remain equally unimportant), on the other hand normalizing positive and negative weights in a shared range makes it easy to compare explanations across predictions.
Most importantly scores represent relative importance in a more consistent way.

To normalize in the _\[-1, 1\]_ range the following code can be used: 

```java

double upper = 1;
double lower = -1;
double max = Arrays.stream(weights).max().orElse(upper);
double min = Arrays.stream(weights).min().orElse(lower);
if (max != min) {
    for (int k = 0; k < weights.length; k++) {
        weights[k] = lower + (weights[k] - min)*(upper - lower) / (max - min);
    }
}

```
This would result in the following normalized saliency scores:

| -0.2424222447398524 | 0.15814772783822595 | -0.642992217317931 | -0.2424222447398524 | -0.2424222447398524 | -0.2424222447398524 | -0.2424222447398524 | -0.2424222447398524 | -0.2424222447398524 | -0.2424222447398524 | -0.2424222447398524 | -0.2831345717454691 | -0.2424222447398524 | -0.2424222447398524 | -0.2424222447398524 | -0.2424222447398524 | -0.2424222447398524 | -0.2424222447398524 | -0.2424222447398524 | -0.2424222447398524 | -1.0 | -0.2859844346358621 | -0.2424222447398524 | -0.2424222447398524 | -0.2424222447398524 | -0.2424222447398524 | 0.15814772783822595 | 0.15814772783822595 | -0.2424222447398524 | -0.2424222447398524 | -0.2424222447398524 | -0.2424222447398524 | 1.0 | -0.2831345717454691 | 0.19886005484384262 | 0.5587177004163046 | -0.2424222447398524 | 

If, on the other hand, we want to mute _adversarial features_ to avoid confusing our users, we can also normalize in the _\[0, 1\]_ range as follows:

```java

double upper = 1;
double lower = 0;
double max = Arrays.stream(weights).max().orElse(upper);
double min = Arrays.stream(weights).min().orElse(lower);
if (max != min) {
    for (int k = 0; k < weights.length; k++) {
        weights[k] = (weights[k] - min) / (max - min);
    }
}
```
Normalizing the saliency scores of the previous explanation in _\[0, 1\]_ would result in the following:

| greetings | from | Tatooine | we | reach | out | from | the | outer | hem | to | announce | you've | just | won | the | rebel | prize | you | will | get | empire | credits | for | free | you | just | need | to | send | us | 100 | $ | for | the | hyperspace | flight |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.3787888776300738 | 0.579073863919113 | 0.1785038913410345 | 0.3787888776300738 | 0.3787888776300738 | 0.3787888776300738 | 0.3787888776300738 | 0.3787888776300738 | 0.3787888776300738 | 0.3787888776300738 | 0.3787888776300738 | 0.35843271412726546 | 0.3787888776300738 | 0.3787888776300738 | 0.3787888776300738 | 0.3787888776300738 | 0.3787888776300738 | 0.3787888776300738 | 0.3787888776300738 | 0.3787888776300738 | 0.0 | 0.35700778268206895 | 0.3787888776300738 | 0.3787888776300738 | 0.3787888776300738 | 0.3787888776300738 | 0.579073863919113 | 0.579073863919113 | 0.3787888776300738 | 0.3787888776300738 | 0.3787888776300738 | 0.3787888776300738 | 1.0 | 0.35843271412726546 | 0.5994300274219213 | 0.7793588502081523 | 0.3787888776300738 | 
