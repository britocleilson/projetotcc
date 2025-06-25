"""
Script criado para simular navegação da loja
Os pesos atribuidos no decorator @task(x) representam a probabilidade de ser executado.
Cada ponto equivale a 5% e está configurado de acordo com as probabilidades definidas no artigo
"""
from locust import HttpUser, task, between
import random

class MicroserviceUser(HttpUser):
    wait_time = between(1, 10)  # Espera entre 1-10s entre as requisições


    def on_start(self):
        self.client.verify = False
        self.products = [
            "0PUK6V6EV0",
            "1YMWWN1N4O",
            "2ZYFJ3GM2N",
            "66VCHSJNUP",
            "6E92ZMYYFZ",
            "9SIQT8TOJO",
            "L9ECAV7KIM",
            "LS4PSXUNUM",
            "OLJCESPC7Z"
        ]

    @task(10)
    def browse_product(self):
        """Navegar por um produto (probabilidade 1/2)"""
        product = random.choice(self.products)
        self.client.get(f"/product/{product}", name="/product/[id]")

    @task(2)
    def change_currency(self):
        """Alterar moeda (probabilidade 1/10)"""
        currencies = ["EUR", "USD", "JPY","GBP","TRY", "CAD"]
        self.client.post("/setCurrency", {
            "currency_code": random.choice(currencies)
        }, name="/setCurrency")

    @task(2)
    def add_to_cart(self):
        """Adicionar ao carrinho (probabilidade 1/5)"""
        product = random.choice(self.products)
        self.client.get(f"/product/{product}", name="/product/[id]")
        self.client.post("/cart", {
            "product_id": product,
            "quantity": random.randint(1,3)
        }, name="/cart")

    @task(4)
    def view_cart(self):
        """Visualizar carrinho (probabilidade 1/5)"""
        self.client.get("/cart", name="/cart")

    @task(1)
    def checkout(self):
        """Fluxo completo de checkout"""
        self.add_to_cart()

        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Referer": f"{self.host}/cart"  # Adicionado para simular navegador
        }

        form_data = {
            "email": "someone@example.com",
            "street_address": "1600 Amphitheatre Parkway",
            "zip_code": "94043",
            "city": "Mountain View",
            "state": "CA",
            "country": "United States",
            "credit_card_number": "4432801561520454",  # Sem hífens
            "credit_card_expiration_month": "1",
            "credit_card_expiration_year": "2026",  # Ano compatível com o HTML
            "credit_card_cvv": "672"
        }

        with self.client.post("/cart/checkout",
                              data=form_data,  # Usar data= em vez de json=
                              headers=headers,
                              name="/cart/checkout",
                              catch_response=True) as response:
            if response.status_code == 422:
                response.failure(f"Erro de validação: {response.text}")

    @task(1)
    def refresh_frontend(self):
        """Atualizar frontend (probabilidade 1/20)"""
        self.client.get("/", name="/")
