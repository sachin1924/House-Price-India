import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 📁 Load dataset
df = pd.read_csv("C:/Users/SACHIN/Downloads/archive (4)/House Price India.csv")


# 1️⃣ Basic Info
print("Dataset shape before cleaning:", df.shape)
print("\nMissing values per column:\n", df.isnull().sum())

# 2️⃣ Drop duplicate rows
df.drop_duplicates(inplace=True)

# 3️⃣ Rename columns for consistency (optional but helpful)
df.rename(columns=lambda x: x.strip().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").lower(), inplace=True)

# 4️⃣ Drop irrelevant columns (e.g., ID, Postal Code, Date)
cols_to_drop = ['id', 'postal_code', 'date']
df.drop(columns=[col for col in cols_to_drop if col in df.columns], inplace=True)

# 5️⃣ Fill or drop missing values
# Example strategy: fill numeric columns with median
for col in df.select_dtypes(include=['float64', 'int64']).columns:
    df[col].fillna(df[col].median(), inplace=True)

# Fill categorical (if any) with mode
for col in df.select_dtypes(include=['object']).columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# 6️⃣ Final check
print("\nDataset shape after cleaning:", df.shape)
print("\nRemaining missing values:\n", df.isnull().sum())

# Optional: Save cleaned data
df.to_csv("Cleaned_House_Price_India.csv", index=False)










# Clean and prepare
df = df.drop_duplicates()
df.rename(columns=lambda x: x.strip().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_").lower(), inplace=True)
df.drop(columns=[col for col in ['id', 'postal_code', 'date'] if col in df.columns], inplace=True)

# Create binary column for renovation status
df['was_renovated'] = df['renovation_year'].apply(lambda x: 1 if x != 0 else 0)

# Rename for clarity
df = df.rename(columns={
    'area_of_the_houseexcluding_basement': 'house_area',
    'area_of_the_basement': 'basement_area'
})

# Only numeric data for correlation
numeric_df = df.select_dtypes(include='number')

# Correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap with Price")
plt.show()

# Pairplot of top 3 features most correlated with price
top_corr = numeric_df.corr()['price'].abs().sort_values(ascending=False)[1:4]
top_features = top_corr.index.tolist()
sns.pairplot(df[top_features + ['price']], diag_kind="kde")
plt.suptitle("Pairplot of Top Features vs Price", y=1.02)
plt.show()

# Pie chart: Renovated vs Not Renovated
renovation_counts = df['was_renovated'].value_counts()
labels = ['Not Renovated', 'Renovated']
plt.figure(figsize=(6, 6))
plt.pie(renovation_counts, labels=labels, autopct='%1.1f%%', colors=['#ff9999','#66b3ff'], startangle=140)
plt.title("Proportion of Renovated vs Non-Renovated Houses")
plt.axis('equal')
plt.show()

# Scatter plot: Living Area Renovated vs Price (if column exists)
if 'living_area_renov' in df.columns:
    sns.scatterplot(x='living_area_renov', y='price', hue='was_renovated', data=df)
    plt.title("Living Area Renovated vs Price")
    plt.show()

# Scatter plots: Longitude and Latitude vs Price
sns.scatterplot(x='longitude', y='price', data=df)
plt.title("Longitude vs Price")
plt.show()

sns.scatterplot(x='lattitude', y='price', data=df)
plt.title("Latitude vs Price")
plt.show()

# Boxplot: Bedrooms vs Price
sns.boxplot(x='number_of_bedrooms', y='price', data=df)
plt.title("Bedrooms vs Price")
plt.show()

# Scatter plot: House Area vs Price
sns.scatterplot(x='house_area', y='price', data=df)
plt.title("House Area vs Price")
plt.show()

# Bar Chart: Average price by number of schools nearby
if 'number_of_schools_nearby' in df.columns:
    avg_price_school = df.groupby('number_of_schools_nearby')['price'].mean()
    sns.barplot(x=avg_price_school.index, y=avg_price_school.values)
    plt.title("Avg Price vs Number of Schools Nearby")
    plt.xlabel("Number of Schools Nearby")
    plt.ylabel("Average Price")
    plt.show()

# Boxplot: Hospitals nearby vs Price
if 'number_of_hospitals_nearby' in df.columns:
    sns.boxplot(x='number_of_hospitals_nearby', y='price', data=df)
    plt.title("Hospitals Nearby vs Price")
    plt.show()