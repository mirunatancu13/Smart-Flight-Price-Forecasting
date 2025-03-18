import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from bs4 import BeautifulSoup
import csv
from datetime import datetime, timedelta
import random
import sys

# aici pentru a retine in ce zi vrem sa achizitionam biletu
data_curenta = datetime.now()
data_curenta_formatata = data_curenta.strftime("%Y-%m-%d")


def scrape_from_kiwi(driver, destination, origin, departure_date, return_date, tickets):
    for _ in range(7):
        try:
            load_more_button = driver.find_element(By.XPATH,
                                                   '//*[@id="react-view"]/div[2]/div[4]/div/div/div/div/div/div[3]/div/div/div[4]/div/div/button/div')
            load_more_button.click()
            time.sleep(5)  #asteptam ca sa se incarce zboruri
        except Exception as e:
            print("Nu am gasit butonul de 'Incarcati mai multe':", e)
            break

    page_source = driver.find_element(By.TAG_NAME, 'html').get_attribute('innerHTML') #toata pagina
    page_soup = BeautifulSoup(page_source, "html.parser")
    cards = page_soup.find_all('div', class_="group/result-card relative cursor-pointer leading-normal")

    print(f"Am gasit {len(cards)} carduri cu tichete.")

    for card in cards:
        price_elements = card.find_all('span', class_=["length-7", "length-9"])

        airline_element = card.find('img',
                                    class_="max-w-none bg-transparent rounded-100 last:self-end h-icon-large w-icon-large")
        airline = airline_element.get('title') if airline_element else 'Nu se stie'#extrage atributul title

        time_elements = card.find_all('time')
        if len(time_elements) >= 4:
            hour1 = time_elements[0].get_text(strip=True)
            hour2 = time_elements[2].get_text(strip=True)
            hour3 = time_elements[3].get_text(strip=True)
            hour4 = time_elements[5].get_text(strip=True)
        else:
            hour1 = hour2 = hour3 = hour4 = 'Nu se stie'

        for price_element in price_elements:
            if price_element:
                price = price_element.get_text(strip=True).replace('\xa0lei', '').replace(' ', '')
                price = price.replace('.', '').replace(' ', '')
                ticket = {
                    'Destinatie': destination,
                    'Origine': origin,
                    'Data_Plecare1': departure_date,
                    'Data_Plecare2': return_date,
                    'Airline': airline,
                    'Pret (lei)': price,
                    'Ora_Departure_1stTrip': hour1,
                    'Ora_Arrival_1stTrip': hour2,
                    'Ora_Departure_2ndTrip': hour3,
                    'Ora_Arrival_2ndTrip': hour4,
                    'Data_Verificare_Bilete': data_curenta_formatata,
                }
                tickets.append(ticket)


def generate_random_flights(destination, origin, departure_date, return_date, min_price, max_price, num_rows=40):
    airlines = ['Ryanair', 'Wizz Air Malta', 'British Airways', 'Lufthansa', 'KLM']
    hours = [f"{random.randint(0, 23):02}:{random.randint(0, 59):02}" for _ in range(num_rows)]
    dates = [(datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)] 

    rows = []
    for i in range(num_rows):
        row = {
            'Destinatie': destination,
            'Origine': origin,
            'Data_Plecare1': departure_date,
            'Data_Plecare2': return_date,
            'Airline': random.choice(airlines),
            'Pret (lei)': random.randint(min_price, max_price),
            'Ora_Departure_1stTrip': random.choice(hours),
            'Ora_Arrival_1stTrip': random.choice(hours),
            'Ora_Departure_2ndTrip': random.choice(hours),
            'Ora_Arrival_2ndTrip': random.choice(hours),
            'Data_Verificare_Bilete': random.choice(dates),
        }
        rows.append(row)

    return rows


def write_to_csv(tickets):
    with open('flights.csv', mode='w', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=[
            'Destinatie', 'Origine', 'Data_Plecare1', 'Data_Plecare2',
            'Airline', 'Pret (lei)', 'Ora_Departure_1stTrip', 'Ora_Arrival_1stTrip',
            'Ora_Departure_2ndTrip', 'Ora_Arrival_2ndTrip', 'Data_Verificare_Bilete',
        ])
        writer.writeheader()
        writer.writerows(tickets)


##pana aici a fost ws----------------------------------------------------------------------------------------------------------------------------------------------
class FlightPriceAnalyzer:
    def __init__(self, csv_path):
        """
        Initialize the Flight Price Analyzer with data loading and preprocessing

        Args:
            csv_path (str): Path to the CSV file containing flight data
        """
        try:
            self.data = pd.read_csv(csv_path)
            self.preprocess_data()
        except Exception as e:
            print(f"Error loading data: {e}")
            self.data = None

    def preprocess_data(self):
        """
        Preprocess the flight data for analysis
        """
        
        time_columns = ['Ora_Departure_1stTrip', 'Ora_Arrival_1stTrip',
                        'Ora_Departure_2ndTrip', 'Ora_Arrival_2ndTrip']

        for col in time_columns:
            self.data[f'{col}_Min'] = self.data[col].apply(self._time_to_minutes)

        
        self.label_encoder = LabelEncoder()
        self.data['Airline_Encoded'] = self.label_encoder.fit_transform(self.data['Airline'])

        
        self.data['Day_Of_Week_1'] = pd.to_datetime(self.data['Data_Plecare1']).dt.dayofweek
        self.data['Days_Until_Departure_1'] = (
                pd.to_datetime(self.data['Data_Plecare1']) - pd.to_datetime(self.data['Data_Verificare_Bilete'])
        ).dt.days

        self.data['Day_Of_Week_2'] = pd.to_datetime(self.data['Data_Plecare2']).dt.dayofweek
        self.data['Days_Until_Departure_2'] = (
                pd.to_datetime(self.data['Data_Plecare2']) - pd.to_datetime(self.data['Data_Verificare_Bilete'])
        ).dt.days

    @staticmethod
    def _time_to_minutes(time_str):
        """
        Convert time in HH:MM format to minutes from midnight

        Args:
            time_str (str): Time in HH:MM format

        Returns:
            float: Minutes from midnight or NaN
        """
        try:
            hours, minutes = map(int, time_str.split(':'))
            return hours * 60 + minutes
        except:
            return np.nan

    def _prepare_data(self, departure_city, arrival_city):
        """
        Prepare data for a specific route

        Args:
            departure_city (str): Origin city
            arrival_city (str): Destination city

        Returns:
            tuple: Filtered and processed features and target
        """
        
        filtered_data = self.data[
            (self.data['Origine'] == departure_city) &
            (self.data['Destinatie'] == arrival_city)
            ].copy()

        if filtered_data.empty:
            print(f"No flights found between {departure_city} and {arrival_city}")
            return None, None

        
        feature_columns = [
            'Airline_Encoded',
            'Ora_Departure_1stTrip_Min',
            'Ora_Arrival_1stTrip_Min',
            'Ora_Departure_2ndTrip_Min',
            'Ora_Arrival_2ndTrip_Min',
            'Day_Of_Week_1',
            'Days_Until_Departure_1',
            'Day_Of_Week_2',
            'Days_Until_Departure_2'
        ]

        
        filtered_data.dropna(subset=feature_columns + ['Pret (lei)'], inplace=True)

        if len(filtered_data) < 10:
            print("Insufficient data for reliable prediction")
            return None, None

        X = filtered_data[feature_columns]
        y = filtered_data['Pret (lei)']

        
        scaler = StandardScaler()
        X_scaled = X.copy()
        numerical_features = [col for col in feature_columns if col != 'Airline_Encoded']
        X_scaled[numerical_features] = scaler.fit_transform(X[numerical_features])

        return X_scaled, y

    def predict_prices(self, departure_city, arrival_city):
        """
        Predict flight prices for a specific route

        Args:
            departure_city (str): Origin city
            arrival_city (str): Destination city

        Returns:
            dict: Prediction results and performance metrics
        """
        
        X, y = self._prepare_data(departure_city, arrival_city)

        if X is None or y is None:
            return None

        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

       
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10]
        }

        model = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3,
                             scoring='neg_mean_absolute_error')
        model.fit(X_train, y_train)

        best_model = model.best_estimator_

        
        y_pred = best_model.predict(X_test)

        results = {
            'mae': mean_absolute_error(y_test, y_pred),
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }

        
        unique_airlines = self.data[
            (self.data['Origine'] == departure_city) &
            (self.data['Destinatie'] == arrival_city)
            ]['Airline'].unique()

        airline_predictions = []
        for airline in unique_airlines:
            airline_sample = X.iloc[0].copy()
            airline_sample['Airline_Encoded'] = self.label_encoder.transform([airline])[0]
            predicted_price = best_model.predict([airline_sample])[0]
            airline_predictions.append({
                'Airline': airline,
                'Predicted_Price': predicted_price
            })

        results['airline_predictions'] = airline_predictions

        
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Prices')
        plt.ylabel('Predicted Prices')
        plt.title(f'Price Predictions: {departure_city} to {arrival_city}')
        plt.show()

        return results

    def generate_comprehensive_report(self, departure_city, arrival_city):
        """
        Generate a detailed flight price analysis report

        Args:
            departure_city (str): Origin city
            arrival_city (str): Destination city

        Returns:
            dict: Comprehensive analysis report
        """
        
        predictions = self.predict_prices(departure_city, arrival_city)

        if predictions is None:
            return None

        
        route_data = self.data[
            (self.data['Origine'] == departure_city) &
            (self.data['Destinatie'] == arrival_city)
            ]

        
        price_insights = {
            'average_price': route_data['Pret (lei)'].mean(),
            'minimum_price': route_data['Pret (lei)'].min(),
            'maximum_price': route_data['Pret (lei)'].max(),
            'price_standard_deviation': route_data['Pret (lei)'].std()
        }

        
        self._create_price_visualizations(route_data)

        
        report = {
            'route': f'{departure_city} to {arrival_city}',
            'price_predictions': predictions,
            'price_insights': price_insights
        }

        return report

    def _create_price_visualizations(self, route_data):
        """
        Create visualizations for price analysis

        Args:
            route_data (pd.DataFrame): Filtered route-specific data
        """
        plt.figure(figsize=(15, 5))

        
        plt.subplot(1, 2, 1)
        sns.histplot(route_data['Pret (lei)'], kde=True)
        plt.title('Price Distribution')
        plt.xlabel('Price (lei)')
        plt.ylabel('Frequency')

        
        plt.subplot(1, 2, 2)
        sns.boxplot(x='Airline', y='Pret (lei)', data=route_data)
        plt.title('Prices by Airline')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def best_booking_times(self):
        """
        Analyze and visualize the best times to book flights for cost savings.
        """
        self.data['Booking_Interval'] = self.data['Days_Until_Departure_1']
        interval_groups = self.data.groupby('Booking_Interval')['Pret (lei)'].mean().reset_index()

        plt.figure(figsize=(10, 6))
        sns.lineplot(data=interval_groups, x='Booking_Interval', y='Pret (lei)')
        plt.title('Average Flight Prices vs. Days Before Departure')
        plt.xlabel('Days Before Departure')
        plt.ylabel('Average Price (lei)')
        plt.grid()
        plt.show()


def ML(dp, ac):
    file_path = 'flights.csv'
    analyzer = FlightPriceAnalyzer(file_path)

    departure_city = ac
    arrival_city = dp

    report = analyzer.generate_comprehensive_report(departure_city, arrival_city)

    if report:
        print("\n--- Comprehensive Flight Price Report ---")
        print(f"Route: {report['route']}")

        print("\nPrice Insights:")
        for key, value in report['price_insights'].items():
            print(f"- {key.replace('_', ' ').title()}: {value:.2f}")

        print("\nAirline Price Predictions:")
        for pred in report['price_predictions']['airline_predictions']:
            print(f"- {pred['Airline']}: {pred['Predicted_Price']:.2f} lei")

        print("\nModel Performance:")
        performance = report['price_predictions']
        print(f"Mean Absolute Error: {performance['mae']:.2f}")
        print(f"Root Mean Squared Error: {performance['rmse']:.2f}")
        print(f"R-squared Score: {performance['r2']:.2f}")

    
    analyzer.best_booking_times()



def main(origin, destination, departure_date, return_date):
    url = f"https://www.kiwi.com/ro/search/results/{origin}/{destination}/{departure_date}/{return_date}"
    options = webdriver.ChromeOptions()
    #options.add_argument("--headless")
    options.add_argument("--start-maximized")
    driver = webdriver.Chrome(options=options)
    driver.get(url)
    print("Se acceseaza pagina...")

    try:
        accept_button = WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'orbit-button-primitive-content') and contains(text(), 'Acceptați')]"))
        )
        accept_button.click()
        time.sleep(5)
        print("Butonul 'Acceptati' a fost apasat.")
    except Exception as e:
        print(f"Eroare la apasarea butonului: {e}")

    tickets = []
    scrape_from_kiwi(driver, destination, origin, departure_date, return_date, tickets)

    if tickets:
        prices = [int(ticket['Pret (lei)']) for ticket in tickets if ticket['Pret (lei)'].isdigit()]
        min_price = min(prices) if prices else 500  #niste valori implicite random
        max_price = max(prices) if prices else 3000

        random_tickets = generate_random_flights(destination, origin, departure_date, return_date, min_price, max_price)
        tickets.extend(random_tickets)

    write_to_csv(tickets)
    driver.quit()

    ML(destination, origin)
