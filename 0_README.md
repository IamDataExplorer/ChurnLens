# ChurnLens: Analysing & Predicting Customer Attrition

Churn happens when the customer stops doing business with the organization or simply loss of customers over a period of time. 

For any business organization, retaining old customers is more cost-efficient than acquiring new ones. Particularly in the banking sector, the importance of retaining customers is even higher. Banks offer many services like savings accounts, credit cards, loans, and investment options that build long-term relationships with customers. When a customer leaves a bank, the institution not only loses future business but also valuable customer data and the potential for cross-selling other services.

To tackle this problem, we have set the following objectives:
#### Add objectives here

We used [Bank Churn Dataset](https://www.kaggle.com/datasets/rangalamahesh/bank-churn/data?select=train.csv) from Kaggle with 14 features and 1,65,033 entries. The features are:
| Column Name       | Description                                                                 |
| :---------------- | :-------------------------------------------------------------------------- |
| id                | Row number                                                                  |
| CustomerId        | Unique ID assigned to each customer                                         |
| Surname           | Surname of the customer                                                     |
| CreditScore       | Customer's credit score (range: 350-850)                                    |
| Geography         | Country of the customer (France, Spain, Germany)                            |
| Gender            | Male or Female                                                              |
| Age               | Customer's age (in years)                                                   |
| Tenure            | Number of years the customer has been with the bank                         |
| Balance           | Bank balance of the customer                                                |
| NumOfProducts     | Number of products (services) the customer has purchased                    |
| HasCrCard         | Whether the customer owns a credit card (0 = No, 1 = Yes)                   |
| IsActiveMember    | Whether the customer is an active member (0 = No, 1 = Yes)                  |
| EstimatedSalary   | Customer's estimated salary                                                 |
| Exited            | Whether the customer left the bank (0 = No, 1 = Yes)                        |

