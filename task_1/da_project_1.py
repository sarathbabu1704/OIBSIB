import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv("Fashion_Retail_Sales.csv", encoding="latin1")
df.drop_duplicates(inplace=True)
df.ffill(inplace=True)

if 'Date Purchase' in df.columns:
    df['Date Purchase'] = pd.to_datetime(df['Date Purchase'])
else:
    print("Warning: 'Date Purchase' column not found. Time series analysis may be limited.")

if 'Sales' not in df.columns and 'Purchase Amount (USD)' in df.columns:
    df['Sales'] = df['Purchase Amount (USD)']
elif 'Sales' in df.columns:
    df['Sales'] = pd.to_numeric(df['Sales'], errors='coerce')
    df.ffill(inplace=True)
else:
    print("Warning: 'Sales' column not found. Sales analysis may be limited.")

# Product Category
if 'Item Purchased' in df.columns:
    df['Product_Category'] = df['Item Purchased']
else:
    print("Warning: 'Item Purchased' column not found. Product analysis may be limited.")
customer_cols = [col for col in df.columns if 'Customer' in col or 'Gender' in col or 'Age' in col]
print("Basic Statistics for Sales:")
print(df['Sales'].describe())  
print("Mode of Sales:")
print(df['Sales'].mode())

if 'Age' in df.columns:
    print("\nCustomer Age Statistics:")
    print(df['Age'].describe())

if 'Date Purchase' in df.columns and 'Sales' in df.columns:
    sales_trend = df.groupby('Date Purchase')['Sales'].sum()
    plt.figure(figsize=(10,5))
    plt.plot(sales_trend.index, sales_trend.values, color='blue', marker='o', linestyle='-')
    plt.title("Daily Sales Trend")
    plt.xlabel("Date")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Monthly sales trend
    df['Month'] = df['Date Purchase'].dt.to_period('M')
    monthly_sales = df.groupby('Month')['Sales'].sum()
    monthly_sales.plot(kind='line', figsize=(10,5), color='green', marker='o')
    plt.title("Monthly Sales Trend")
    plt.xlabel("Month")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("Cannot plot sales trends due to missing data.")

# Product Analysis
if 'Product_Category' in df.columns:
    product_sales = df.groupby('Product_Category')['Sales'].sum().sort_values(ascending=False)
    plt.figure(figsize=(10,5))
    sns.barplot(x=product_sales.index, y=product_sales.values, palette='viridis')
    plt.title("Sales by Product Category")
    plt.xlabel("Product Category")
    plt.ylabel("Total Sales")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Customer Analysis
if 'Gender' in df.columns:
    gender_sales = df.groupby('Gender')['Sales'].sum()
    plt.figure(figsize=(6,4))
    sns.barplot(x=gender_sales.index, y=gender_sales.values, palette='pastel')
    plt.title("Sales by Gender")
    plt.ylabel("Total Sales")
    plt.tight_layout()
    plt.show()

if 'Age' in df.columns:
    plt.figure(figsize=(8,4))
    sns.histplot(df['Age'], bins=15, kde=True, color='purple')
    plt.title("Customer Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()
plt.figure(figsize=(6,4))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

print("\n--- Recommendations ---")
if 'Product_Category' in df.columns:
    top_products = product_sales.head(3).index.tolist()
    print(f"Focus on top-selling products: {top_products}")

if 'Month' in df.columns:
    peak_month = monthly_sales.idxmax()
    print(f"Stock and promote more during peak month: {peak_month}")

if 'Gender' in df.columns:
    top_gender = gender_sales.idxmax()
    print(f"Target marketing campaigns to: {top_gender} customers")
