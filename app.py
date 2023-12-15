import pytesseract
from flask import Flask, request, jsonify
from PIL import Image
import base64
import re
from dateutil import parser
import spacy
from spacy.matcher import Matcher
from io import BytesIO

from transformers import MarianMTModel, MarianTokenizer

app = Flask(__name__)

# Carrega o modelo e tokenizer para tradução de texto
model_name = "Helsinki-NLP/opus-mt-tc-big-en-pt"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Carrega o modelo em português do spaCy
nlp = spacy.load("pt_core_news_sm")

def extract_text_from_image(image_base64):
    try:
        # Converte a imagem Base64 para uma imagem PIL
        image_data = Image.open(BytesIO(base64.b64decode(image_base64)))

        # Converte a imagem para RGB antes de usar o pytesseract
        image_data = image_data.convert('RGB')

        # Usa o pytesseract para extrair texto da imagem
        extracted_text = pytesseract.image_to_string(image_data, lang='por')

        return extracted_text
    except Exception as e:
        print(f"Erro ao extrair texto da imagem: {str(e)}")
        return ""

def translate_text(text, model, tokenizer):
    # Tokeniza e traduz o texto
    input_ids = tokenizer.encode(text, return_tensors="pt")
    output_ids = model.generate(input_ids)
    translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return translated_text

def extract_payment_details(translated_text):
    payment_data = {
        'data_horario': None,
        'destino': None,
        'origem': None,
        'valor': None,
    }

    try:
        date_match = re.search(r'(\d{1,2}\/\d{1,2}\/\d{4}(?:[^0-9]\d{1,2}:\d{1,2}(?::\d{1,2})?)?)', translated_text)
        if date_match:
            payment_data['data_horario'] = str(parser.parse(date_match.group(1)))

        value_match = re.search(r'R\$\s*(\d+,\d{2})', translated_text)
        if value_match:
            payment_data['valor'] = value_match.group(1)

        doc = nlp(translated_text)

        # Configura o Matcher
        matcher = Matcher(nlp.vocab)

        # Padrão para encontrar nomes de pessoas
        pattern_person = [{"label": "PER"}]
        matcher.add("PERSON", [pattern_person])

        # Padrão para encontrar nomes de organizações
        pattern_organization = [{"label": "ORG"}]
        matcher.add("ORGANIZATION", [pattern_organization])

        matches = matcher(doc)

        dest_name = None
        orig_name = None

        # Extrai nomes com base nos matches encontrados
        for match_id, start, end in matches:
            if doc[start:end].label_ == "PERSON":  # Pessoa
                if dest_name is None:
                    dest_name = doc[start:end].text.strip()
                else:
                    orig_name = doc[start:end].text.strip()
            elif doc[start:end].label_ == "ORGANIZATION":  # Organização
                if dest_name is None:
                    dest_name = doc[start:end].text.strip()
                else:
                    orig_name = doc[start:end].text.strip()

        payment_data['destino'] = dest_name if dest_name else None
        payment_data['origem'] = orig_name if orig_name else None

    except Exception as e:
        print(f"Erro ao identificar detalhes de pagamento: {str(e)}")

    return payment_data

@app.route('/process_payment_receipt', methods=['POST'])
def process_payment_receipt():
    try:
        # Obtém a imagem em formato Base64 da requisição
        image_base64 = request.json['image_base64']

        # Extrai texto da imagem
        extracted_text = extract_text_from_image(image_base64)
        print(f"Texto extraído da imagem: {extracted_text}")

        # Extrai detalhes do pagamento a partir do texto
        translated_text = translate_text(extracted_text, model, tokenizer)
        payment_details = extract_payment_details(translated_text)

        # Armazena os dados e retorna a resposta
        # ... (Você pode usar um banco de dados para armazenar os dados)

        return jsonify({'success': True, 'data': payment_details})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)