# Luppa AI teste de Andre Emidio

Projeto para teste backend para a empresa Luppa AI

Objetivos : Criar uma API para receber uma imagem em base64 e inferir que tipo de produto é, soja ou arvore.

Para rodar o projeto é necessário dar clone no repositório, e criar um ambiente virtual com [*
Pipenv*.](https://pipenv.pypa.io/en/latest/)

Para criar o ambiente é necessário rodar o comando.

````pipenv shell````

Para instalar as dependências

````pipenv install````

Para rodar o projeto localmente.

````python app.py````

A rota para fazer a predição é

````/api/predict````

O body para teste é em formato json

````
{
	"image":"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"
}
````