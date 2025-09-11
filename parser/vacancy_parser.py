import polars as pl
import requests
import time
import random
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

class Vacancy_parser:
    def __init__(self, hh_api_token: str, path: str = "vacancy.patquet"):
        self.headers = {
            'Authorization': f'Bearer {hh_api_token}'
        }
        self.path = path

    def get_vacancies(self, city, vacancy, page):
        url = 'https://api.hh.ru/vacancies'
        params = {
            'text': f"{vacancy}",
            'per_page': 100,
            'page': page
        }
    
        response = requests.get(url, params=params, headers=self.headers)
        response.raise_for_status()
        return response.json()


    def get_vacancy_skills(self, vacancy_id: str) -> List[str]:
    """Получает навыки для конкретной вакансии"""
        try:
            url = f"https://api.hh.ru/vacancies/{vacancy_id}"
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            data = response.json()
    
            skills = data.get('key_skills', [])
            if skills and isinstance(skills, list):
                if isinstance(skills[0], dict) and 'name' in skills[0]:
                    return [skill['name'] for skill in skills]
                elif isinstance(skills[0], str):
                    return skills
            return []
        except Exception as e:
            logging.error(f"Ошибка при получении навыков для вакансии {vacancy_id}: {e}")
            return []


    def get_industry(self, company_id):
        if company_id is None:
            return 'Unknown'
    
        url = f'https://api.hh.ru/employers/{company_id}'
        response = requests.get(url, headers=self.headers)
        if response.status_code == 404:
            return 'Unknown'
        response.raise_for_status()
        data = response.json()
    
        if 'industries' in data and len(data['industries']) > 0:
            return data['industries'][0].get('name')
        return 'Unknown'

    def parse_vacancies_incremental(self, city_id: int = 1, city: str = "Москва") -> pl.DataFrame:
        """Версия с постепенным наращиванием DataFrame"""
        vacancies = [
            'Computer vision',
            'Data Analyst', 'Data Engineer', 'Data Science', 'Data Scientist', 'ML Engineer',
            'MLOps инженер', 'AI',
            'Product Manager', 'Python Developer', 'Web Analyst', 'Аналитик данных',
            'Бизнес-аналитик', 'Системный аналитик', 'Финансовый аналитик', 'ML',
            'Deep Learning', 'NLP', 'LLM', "Project Manager", 'Product Owner','Time series',
        ]
    
        df_final = None
        cities = {
            'Москва': 1,
        }
        for city, city_id in cities.items():
            for vacancy in vacancies:
                page = 0
                vacancy_data = []
                logging.info(f"Парсинг вакансии: {vacancy}")
    
                while True:
                    try:
                        data = get_vacancies(city_id, vacancy, page)
                        if page==0:
                            logging.info(f"Число страниц по запросу {data.get('pages')}")
                        if not data.get('items'):
                            break
    
                        for item in data['items']:
                            salary_data = item.get('salary')
                            if salary_data is None:
                                salary_from = salary_to = salary_currency = None
                                salary_str = "з/п не указана"
                            else:
                                salary_from = salary_data.get('from')
                                salary_to = salary_data.get('to')
                                salary_currency = salary_data.get('currency')
                                salary_str = f"{salary_from or ''}-{salary_to or ''} {salary_currency or ''}"
    
                            skills_list = get_vacancy_skills(item['id'])
                            industry = get_industry(item['employer'].get('id'))
    
                            work_format = item.get('work_format', [])
                            work_format_ids = [fmt['id'] for fmt in work_format]
                            work_format_names = [fmt['name'] for fmt in work_format]
    
                            record = {
                                'vacancy_id': item['id'],
                                'title': item['name'],
                                'loc': item['area']['name'],
                                'requirement': item['snippet'].get('requirement', ''),
                                'responsibility': item['snippet'].get('responsibility', ''),
                                'work_format_ids': work_format_ids,
                                'work_format_names': work_format_names,
                                'skills': skills_list,
                                'company': item['employer']['name'],
                                'industry': industry,
                                'experience': item['experience'].get('name', 'Не указан'),
                                'salary_from': salary_from,
                                'salary_to': salary_to,
                                'salary_currency': salary_currency,
                                'salary_str': salary_str,
                                'url': item['alternate_url'],
                                'published_at': item.get('published_at'),
                                'source_vacancy': vacancy
                            }
    
                            vacancy_data.append(record)
    
                        if page >= data.get('pages'):
                            logging.info(f"Страницы {page} из {data.get('pages')} закончились")
                            break
    
                        page += 1
    
                    except requests.HTTPError as e:
                        logging.error(f"Ошибка при обработке вакансии {vacancy}, страница {page}: {e}")
                        break
                time.sleep(random.uniform(1, 2))
                if vacancy_data:
                    df_current = pl.DataFrame(vacancy_data)
    
                    if df_final is None:
                        df_final = df_current
                    else:
                        df_final = pl.concat([df_final, df_current])
    
        if df_final is not None:
            df_final = df_final.with_columns([
                pl.col('published_at').str.strptime(pl.Datetime, format='%Y-%m-%dT%H:%M:%S%z'),
                pl.col('salary_from').cast(pl.Int64),
                pl.col('salary_to').cast(pl.Int64),
            ]).unique(subset=['vacancy_id'])
            df_final = df_final.with_columns(pl.col('vacancy_id').cast(pl.Int64).alias('vacancy_id'))
            logging.info(f"Спаршено {len(df_final)} уникальных вакансий")
            
            df_final.write_parquet(self.path)
            logging.info(f"Датасет сохранён в {self.path}")
            return df_final
        else:
            return pl.DataFrame()

if __name__ == '__main__':
    HH_API_TOKEN=""
    parser = Vacancy_parser(hh_api_token=HH_API_TOKEN)
    parser.parse_vacancies_incremental()