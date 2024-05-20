library(dplyr)
library(ggplot2)
library(purrr)
library(tidyr)
library(class)
library(caTools)
library(tibble)
housing_data = read.csv("HousingData.csv")

str(housing_data)
summary(housing_data)

housing_data %>% map(~sum(is.na(.)))

housing_data = housing_data %>%
  replace_na(list(CRIME_RATE = 0, RESIDENTIAL_LANDS = 0, NON_RETAIL = 0, CHARLES_RIVER = 0, OCCUPIED_UNITS = 0, LOWER_STATUS = 0))

dim(housing_data)

# -------------------------------------------- Data Visualization --------------------------------------------
ggplot(housing_data, aes(x = WEIGHTED_DISTANCES, y = RADIAL_HIGHWAYS)) +
  geom_point(size=3, alpha=0.3, color = "red") +
  geom_smooth(method = "lm") +
  theme_minimal() +
  labs(title = "Pairplot of Housing Dataset")

ggplot(housing_data, aes(x = MED_VALUE)) +
  geom_histogram(binwidth = 5, fill = "violetred", color = "black") +
  theme_minimal() +
  labs(title = "Histogram of Median House Values")

# -------------------------------------------- Normalization --------------------------------------------
range(housing_data$NOX)

# Simple Scaling
simple_scale = housing_data$NOX / max(housing_data$NOX)
head(simple_scale)

# Min-Max Scaling
minmax_scale = (housing_data$NOX - min(housing_data$NOX)) / (max(housing_data$NOX) - min(housing_data$NOX))
head(minmax_scale)

# Z-Score Scaling
z_scale = (housing_data$NOX - mean(housing_data$NOX)) / sd(housing_data$NOX)
head(z_scale)

# -------------------------------------------- ANOVA --------------------------------------------
avg_med_value = housing_data %>%
  group_by(RADIAL_HIGHWAYS) %>%
  summarize(avg_med_value = mean(MED_VALUE, na.rm = TRUE))

ggplot(avg_med_value, aes(x = RADIAL_HIGHWAYS, y = avg_med_value)) +
  geom_bar(stat = "identity", fill = "skyblue", color = "black") +
  labs(title = "Average MEDIAN VALUE by RADIAL HIGHWAYS",
       x = "RADIAL HIGHWAYS",
       y = "Average MEDIAN VALUE")

housing_subset = housing_data %>%
  select(MED_VALUE, RADIAL_HIGHWAYS) %>%
  filter(!is.na(RADIAL_HIGHWAYS))

housing_aov = aov(MED_VALUE ~ RADIAL_HIGHWAYS, data = housing_subset)
summary(housing_aov)

# -------------------------------------------- Correlation --------------------------------------------
# Pearson Correlation
housing_data %>%
  select(MED_VALUE,ROOM_NUMBER) %>%
  cor(method = "pearson")

housing_data %>% cor.test(~MED_VALUE + ROOM_NUMBER, data = .)

correlation = cor(housing_data$MED_VALUE, housing_data$ROOM_NUMBER)
print(correlation)

correlation_matrix = cor(housing_data[, sapply(housing_data, is.numeric)])

# Positive Linear Relationship
ggplot(housing_data, aes(x = ROOM_NUMBER, y = MED_VALUE)) +
  geom_point() +
  geom_smooth(method = "lm", color = "blue") +
  labs(title = "Relationship between ROOM NUMBER and MEDIAN VALUE", x = "ROOM NUMBER", y = "MEDIAN VALUE")

# Weak Positive Linear Relationship
ggplot(housing_data, aes(x = CHARLES_RIVER, y = MED_VALUE)) +
  geom_point() +
  geom_smooth(method = "lm", color = "blue") +
  labs(title = "Relationship between CHARLES RIVER and MEDIAN VALUE", x = "CHARLES RIVER", y = "MEDIAN VALUE")

# Negative Linear Relationship
ggplot(housing_data, aes(x = LOWER_STATUS, y = MED_VALUE)) +
  geom_point() +
  geom_smooth(method = "lm", color = "red") +
  labs(title = "Relationship between LOWER STATUS and MEDIAN VALUE", x = "LOWER STATUS", y = "MEDIAN VALUE")

# Weak Negative Linear Relationship
ggplot(housing_data, aes(x = TAX, y = MED_VALUE)) +
  geom_point() +
  geom_smooth(method = "lm", color = "red") + 
  labs(title = "Relationship between TAX and MEDIAN VALUE", x = "TAX", y = "MEDIAN VALUE")

# -------------------------------------------- Regression --------------------------------------------
# Simple Linear Regression (SLR)
linear_model = lm(MED_VALUE ~ TAX, data = housing_data)
summary(linear_model)

new_data = data.frame(TAX = c(300, 320, 340))
prediction = predict(linear_model, newdata = new_data, interval = "confidence")
print(prediction)

# Multiple Linear Regression (MLR)
mlr = lm(MED_VALUE ~ ROOM_NUMBER + TAX, data = housing_data)
summary(mlr)

multi_data = data.frame(ROOM_NUMBER = c(5, 6, 7), TAX = c(300, 320, 340))
pred = predict(mlr, newdata = multi_data, interval = "confidence")
print(pred)

# -------------------------------------------- K-Means Clustering --------------------------------------------
housing_2_columns = housing_data[, c("MED_VALUE", "ROOM_NUMBER")]
housing_2_columns = scale(housing_2_columns)
head(housing_2_columns)

# Elbow Method
set.seed(123)
n = 10
results = numeric(n)
for (i in 1:n) {
  km.out = kmeans(housing_2_columns, centers = i, nstart = 20)
  results[i] = km.out$tot.withinss
}

result_df = tibble(clusters = 1:n, results = results)

result_plot = ggplot(result_df, aes(x = clusters, y = results)) +
  geom_point() +
  geom_line() +
  xlab("Number of clusters") +
  ylab("Total within cluster sum of squares") +
  ggtitle("Elbow Method for Determining Optimal Number of Clusters")

print(result_plot)

km.out = kmeans(housing_2_columns, centers = 3, nstart = 20)
housing_2_columns = as.data.frame(housing_2_columns)
housing_2_columns$clusterid = factor(km.out$cluster)

ggplot(housing_2_columns, aes(x = ROOM_NUMBER, y = MED_VALUE, color = clusterid)) +
  geom_point() +
  xlab("Room Number") +
  ylab("Median Value") +
  ggtitle("K-Means Clustering of Housing Data")

# -------------------------------------------- K-Nearest Neighbors (KNN) --------------------------------------------
housing_data = na.omit(housing_data)

set.seed(123)
split = sample.split(housing_data$MED_VALUE, SplitRatio = 0.75)
train = subset(housing_data, split == TRUE)
test = subset(housing_data, split == FALSE)

median_value = median(train$MED_VALUE)
train$MED_VALUE = ifelse(train$MED_VALUE > median_value, 1, 0)
test$MED_VALUE = ifelse(test$MED_VALUE > median_value, 1, 0)

train_scaled = scale(train[,-which(names(train) == "MED_VALUE")])
test_scaled = scale(test[,-which(names(test) == "MED_VALUE")])

train_scaled = as.data.frame(train_scaled)
test_scaled = as.data.frame(test_scaled)
train_scaled$MED_VALUE = train$MED_VALUE
test_scaled$MED_VALUE = test$MED_VALUE

# KNN
k = 6
test_pred = knn(train = train_scaled[,-ncol(train_scaled)], 
                 test = test_scaled[,-ncol(test_scaled)], 
                 cl = train_scaled$MED_VALUE, 
                 k = k)

actual = test_scaled$MED_VALUE
cm = table(actual, test_pred)
acc = sum(diag(cm)) / length(actual)
sprintf("Accuracy: %.2f%%", acc * 100)