## OUTLINE
1. Dataset Walkthrough

2. Understanding Dataset Hierarchy

3. Data Preprocessing

4. Exploratory Data Analysis

5. Data Visualization

6. Making Recommendation System


## PROJECT DESCRIPTION

The primary aim of analyzing the Amazon Sales Dataset is delve into product categories, prices, ratings, and sales patterns to identify characteristics that resonate with consumers and propel them to purchase as well as provide actionable recommendations that optimize product development, inform marketing strategies and boost your competitive edge.

Exploring this dataset involves a step-by-step process,where we will clean and prepare the data to ensure it's accuracy and consistency. Followed by summarizing the data using descriptive statistics, visualize the data with charts and graphs to see patterns and relationships. We detect outliers, which are unusual data points, and test our assumptions about the data. We divide the data into groups for better understanding and finally, we summarize our findings.


## ABOUT DATASET

This dataset is having the data of 1K+ Amazon Product's Ratings and Reviews as per their details listed on the official website of Amazon

DATASET URL
https://www.kaggle.com/datasets/karkavelrajaj/amazon-sales-dataset

FEATURES

product_id - Product ID

product_name - Name of the Product

category - Category of the Product

discounted_price - Discounted Price of the Product

actual_price - Actual Price of the Product

discount_percentage - Percentage of Discount for the Product

rating - Rating of the Product

rating_count - Number of people who voted for the Amazon rating

about_product - Description about the Product

user_id - ID of the user who wrote review for the Product

user_name - Name of the user who wrote review for the Product
review_id - ID of the user review

review_title - Short review

review_content - Long review

img_link - Image Link of the Product

product_link - Official Website Link of the Product

## IMPORT LIBRARIES

We will use the following libraries
- Pandas: Data manipulation and analysis
- Numpy: Numerical operations and calculations
- Matplotlib: Data visualization and plotting
- Seaborn: Enhanced data visualization and statistical graphics
- Plotly: Interactive and enhanced data visualization plotting
- Scipy: Scientific computing and advanced mathematical operations

## QUESTIONS

1. What is the average rating for each product category?

2. What is the relationship between discounted price and rating?

3. What rating has the highest rating count?

4. What discount percentage had the highest sales count?

5. What products received the best and worst review?

6. What are the Top 5 categories based with highest ratings?

7. Hypothesis: With Ratings below 3.5, the users showed dissatisfaction in the products.

## THE ANALYSIS

### OBSERVATION 

There are 1465 rows and 16 columns in the dataset.

The data type of all columns is object.

The columns in the datasets are:
'product_id', 'product_name', 'sub_category', 'discounted_price', 'actual_price', 'discount_percentage', 'rating', 'rating_count', 'about_product', 'user_id', 'user_name', 'review_id', 'review_title', 'review_content', 'img_link', 'product_link'

Rating has one unsual character, which will to changed to na(none)

There are a few missing values in the dataset, and they will be dropped.

Category column had to be renamed to properly represent its values and a new column was created to narrow down product category.

The data type for discounted price, actual price, discounted percentage, rating and rating count has to changed into an integer or float for anaylsis purpose.

**Check codes out here : [python_project](/Python_project-amazon.dataset--1/python_project(amazon.dataset).ipynb)**

### 1. The Average Rating For Each Product Category

I created a new column to narrow down product category by splitting up the variables in the original column(category).

```
df3 = (df[['category', 'rating']].groupby('category').agg({'rating':'mean'})).reset_index()
df3 = df3.sort_values(by='rating', ascending = False)
df3
```

The output shows that most product categories generally have a positive customer feedback, with average ratings above 3.50.


| Category                | Rating   |
|-------------------------|----------|
| OfficeProducts          | 4.309677 |
| Toys & Games            | 4.300000 |
| Home Improvement        | 4.250000 |
| Computers & Accessories | 4.155654 |
| Electronics             | 4.081749 |
| Home & Kitchen          | 4.040716 |
| Health & Personal Care  | 4.000000 |
| Musical Instruments     | 3.900000 |
| Car & Motorbike         | 3.800000 |

*Table 1: Product Category rating table*


### 2. The Relationship Between Discounted Price And Rating

The correlation between discounted price and rating is used to determine the relationship between them.

```
correlation_coefficient = df["discounted_price"].corr(df["rating"])
correlation_coefficient
```
0.1211318752606628

Discounted price and rating have a weak positive correlation of 0.12. This means that products with higher discounted prices tend to have slightly higher ratings, but the relationship is not very strong.

### 3. Rating With The Highest Rating Count

A new column named as 'rating_Group' was created and used for this analysis. Bins were created and assigned to the new column.

```
# Define bin edges
bins = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

# Define bin labels
labels = ['2.0-2.5', '2.6-3.0', '3.1-3.5', '3.6-4.0', '4.1-4.5', '4.6-5.0']

# Create a new column 'rating_Group' with the assigned bins
df['rating_Group'] = pd.cut(df['rating'], bins=bins, labels=labels, include_lowest=True)

result=df[['rating_Group','rating_count']].\
groupby(['rating_Group']).agg('sum')
result.reset_index(inplace=True)

fig=px.bar(result,x='rating_Group',y='rating_count',
          labels={'rating_Group':'Grouped Ratings','rating_count':'Rating Count'},
          title='Distribution of rating count across Grouped Ratings')
fig.show()
```
The output shows that majority of the users rate product from 4.1 to 4.5. This also symbolises a positive customer experience 

![Rating Count Accros Group Rating](/images/q4.image.png)
*Bar graph visualizing the distribution of rating count across group ratings*

### 4. Discount Percentage With The Highest Sales Count
```
# Creating a new column to Discount_percentage into range.
# Define bin edges
bins = [0, 25, 50, 75, 100]

# Define bin labels
labels = ['0-25%', '26-50%', '51-75%', '76-100%']

# Create a new column 'Percentage_Group' with the assigned bins

df['Percentage_Group'] = pd.cut(df['discount_percentage'], bins=bins, labels=labels, include_lowest=True)

# Using the grouped discount percentage
result1=df[['Percentage_Group','sub_category']].\
groupby(['Percentage_Group']).agg('count')
result1.reset_index(inplace=True)


fig=px.bar(result1,x='Percentage_Group',y='sub_category',
          labels={'Percentage_Group':'Grouped Discount Percentages','category':'sub category Count'},
          title='Distribution of Product sub Category across Grouped Discount Percentages')
fig.show()
``` 

It was realised that majority of the sales came from products with discount percentage 51-75%

![Highest Discount Percentage Range](/images/q5.image.png)
*Bar graph visualizing the distribution of of product category across discount percentages(grouped)*

### 5. Products With The Best And Worst Review
```
# Calculate sentiment score for each review
df["sentiment"] = df["review_content"].apply(lambda text: TextBlob(text).sentiment.polarity)

# Aggregate sentiment scores by product ID
product_sentiment = df.groupby("product_id")["sentiment"].mean()

# Find the product with the best and worst reviews
best_product = product_sentiment.idxmax()
worst_product = product_sentiment.idxmin()

print("Product with the best review:", best_product)
print("Product with the worst review:", worst_product)
```
```
Product with the best review: B016MDK4F4
Product with the worst review: B09XJ1LM7R
```
Product with the best sentiment score is "Ok,Quality perfect , perfect 5m, must buy", (Product_id = B016MDK4F4) with a scrore of 1.0. This indicates a strong posiitve sentiment.

Product with the worst sentiment score is "tv on off not working, so difficult to battery charge", (Product_id = B09XJ1LM7R ) with a score of -0.6. This indicates a strong negative sentiment. 

Products with negative sentiment scores suggest potential areas for improvement. Further analysis of these products could help identify reasons for these bad reviews and identify potential solutions.


### 6. Top 5 Categories Based On Highest Ratings

```
# Group data by sub_category and calculate average rating
average_ratings = df.groupby("sub_category")["rating"].mean().reset_index()

# Sort by average rating in descending order
average_ratings = average_ratings.sort_values(by="rating", ascending=False)

# Print the top 5 categories
print("Top 5 Product sub categories with highest average ratings:")
for i in range(5):
    sub_category = average_ratings.iloc[i]["sub_category"]
    average_rating = average_ratings.iloc[i]["rating"]
    print(f"{i+1}. {sub_category}: {average_rating:.2f}")

```
```
Top 5 Product sub categories with highest average ratings:
1. Computers&Accessories|Tablets: 4.60
2. Computers&Accessories|NetworkingDevices|NetworkAdapters|PowerLANAdapters: 4.50
3. Electronics|Cameras&Photography|Accessories|Film: 4.50
4. Electronics|HomeAudio|MediaStreamingDevices|StreamingClients: 4.50
5. OfficeProducts|OfficeElectronics|Calculators|Basic: 4.50
```
The top 5 categories have average ratings between 4.50 and 4.60, indicating overall positive customer satisfaction within these areas.

Most of the top-rated categories fall within technology-related domains, including tablets, networking devices, photography accessories, media streaming devices, and calculators.

Within broader categories like "Computers & Accessories" and "Electronics," specific subcategories emerge as particularly well-rated, such as tablets, powerline adapters, film 
accessories, and streaming clients.

Four categories share a rating of 4.50, suggesting similar levels of customer satisfaction across these areas.

The presence of "Basic Calculators" in the top 5 suggests that even relatively simple products can achieve high ratings if they meet customer needs effectively.

### 7. Hypothesis: With Ratings below 3.5, the users showed dissatisfaction in the products.
```
# Define a threshold for classifying reviews
threshold = 0.2

# Add a new column with labels (Good or Bad)
df['sentiment_label'] = df['sentiment'].apply(lambda x: 'Good' if x >= threshold else 'Bad')

# Filter rows with ratings below 3.5
filtered_df = df[df['rating'] < 3.5]

# Count elements in the 'sentiment' column
sentiment_label_count = filtered_df['sentiment_label'].value_counts()

# Check if the count of 'Good' is greater than the count of 'Bad'
satisfied = sentiment_label_count.get('Good', 0) > sentiment_label_count.get('Bad', 0)

# Display the sentiment label count, and conclusion
print(sentiment_label_count)
print("\nConclusion:")

if satisfied:
    print("The user was satisfied with the product.")
else:
    print("The user was not satisfied with the product.")
```
```
sentiment_label
Bad     28
Good    13
Name: count, dtype: int64

Conclusion:
The user was not satisfied with the product
```
The results reveals that at a threshold of 0.2, the users who rated below 3.5 showed dissatisfaction in the products. 

## CONCLUSION

This analysis of the Amazon Sales Dataset provided critical insights into consumer preferences, product performance, and potential opportunities for business strategy enhancement. By examining key metrics like average ratings, review counts, and sentiment scores, we identified trends that underline consumer satisfaction and dissatisfaction.

Our findings showed that most product categories generally receive favorable feedback, with Office Products, Toys & Games, and Home Improvement emerging as top-rated categories. Products with higher review counts reflect significant consumer engagement, often correlating with higher ratings, suggesting popularity and reliability.

The analysis also highlighted that discounts of 51-75% are most effective at driving sales volume, presenting a clear opportunity for marketing strategies. Additionally, while there is a weak positive correlation between discounted price and product rating, it indicates that pricing strategies should not solely rely on discounts to enhance product appeal.

Furthermore, sentiment analysis revealed the importance of addressing negative reviews to improve customer experience. Products with negative feedback could benefit from targeted improvements to mitigate dissatisfaction. The hypothesis testing reinforced that users who rated products below 3.5 often had negative sentiments, emphasizing a clear correlation between lower ratings and user dissatisfaction.

These insights can guide actionable recommendations for optimizing product development, adjusting pricing strategies, and improving marketing efforts to align more closely with consumer preferences and boost competitive advantage.
