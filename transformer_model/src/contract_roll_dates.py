from datetime import datetime, timedelta

def get_third_friday(year, month):
    # Start from the 15th of the month to ensure we find the third Friday
    date = datetime(year, month, 15)
    # Calculate the weekday (0=Monday, ..., 6=Sunday)
    weekday = date.weekday()
    # Calculate how many days to add to reach Friday (4)
    days_until_friday = (4 - weekday) % 7
    # Add days to reach the third Friday
    third_friday = date + timedelta(days=days_until_friday)
    return third_friday

def calculate_expiration_dates(start_date, end_date):
    expiration_dates = []
    current_year = start_date.year
    
    while current_year <= end_date.year:
        for month in [3, 6, 9, 12]:  # March, June, September, December
            expiration_date = get_third_friday(current_year, month)
            if start_date <= expiration_date <= end_date:
                expiration_dates.append(expiration_date)
        current_year += 1
    
    return expiration_dates

# Example usage:
start_date = datetime(2017, 1, 1)
end_date = datetime(2024, 12, 31)

expiration_dates = calculate_expiration_dates(start_date, end_date)

for date in expiration_dates:
    print(f"'{date.strftime("%Y-%m-%d")}'::DATE, ")
