from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

# Dataset informasi untuk platform (diperluas untuk contoh)
faq = {
    "apa itu mamacare?": "mamacare adalah aplikasi buatan putra fuazan fatah kang , ia dikenal sebagai kang refferal",

    # another data 
    "apa itu leverage di bybit dan bagaimana cara kerjanya?": "Leverage memungkinkan Anda untuk membuka posisi lebih besar dari modal yang dimiliki. Misalnya, leverage 10x memungkinkan Anda mengendalikan $1000 hanya dengan $100 margin.",
    
    "berapa leverage maksimal yang tersedia di bybit?": "Leverage maksimal tergantung pada pasangan trading. Untuk BTC/USDT biasanya hingga 100x, sedangkan untuk altcoin umumnya lebih rendah.",
    
    "bagaimana cara menghindari likuidasi di bybit?": "Untuk menghindari likuidasi, gunakan leverage dengan bijak, pasang stop loss, dan selalu pantau rasio margin Anda.",
    
    "apakah bybit aman digunakan?": "Bybit memiliki standar keamanan tinggi seperti 2FA, cold wallet untuk penyimpanan aset, dan sistem pemantauan aktivitas mencurigakan.",
    
    "apa itu isolated dan cross margin di bybit?": "Isolated margin berarti hanya margin di posisi tersebut yang dipertaruhkan, sedangkan cross margin menggunakan seluruh saldo di akun Anda sebagai margin.",
    
    "bagaimana cara membuka posisi trading di bybit?": "Pilih pasangan trading, pilih jenis order (limit/market), masukkan jumlah dan leverage, lalu klik 'Open Long' atau 'Open Short'.",
    
    "apa yang harus dilakukan jika akun saya terkunci?": "Jika akun terkunci, Anda bisa menghubungi customer support Bybit melalui email resmi atau live chat di website untuk proses verifikasi identitas.",
    
    "bagaimana cara menggunakan fitur stop loss dan take profit di bybit?": "Saat membuka posisi, Anda bisa langsung mengatur stop loss dan take profit. Atau, Anda bisa mengaturnya setelah posisi dibuka melalui tab 'Positions'.",
    
    "apakah bybit memiliki aplikasi mobile?": "Ya, Bybit tersedia untuk Android dan iOS. Anda bisa mengunduhnya dari Google Play Store atau App Store untuk melakukan trading di mana saja.",
}

# Inisialisasi OpenAI client
openai_client = OpenAI(api_key='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')

class HybridFAQChatbot:
    def __init__(self, model_name="intfloat/e5-large-v2", threshold=0.85):
        """
        Inisialisasi chatbot dengan model embedding dan threshold kemiripan
        
        Args:
            model_name (str): Nama model sentence transformer yang akan digunakan
            threshold (float): Nilai ambang batas kemiripan (0-1)
        """
        print(f"Memuat model {model_name}...")
        self.model = SentenceTransformer(model_name, device='cpu')
        self.threshold = threshold
        
        # Precompute embeddings untuk semua pertanyaan FAQ
        self.faq_questions = list(faq.keys())
        self.faq_answers = list(faq.values())
        
        # Preprocess pertanyaan dengan format yang diharapkan oleh e5
        preprocessed_questions = [f"query: {q}" for q in self.faq_questions]
        self.faq_embeddings = self.model.encode(preprocessed_questions, convert_to_tensor=True)
        
        print("Chatbot siap digunakan!")
    
    def get_openai_response(self, query):
        """
        Mendapatkan respons dari OpenAI API
        
        Args:
            query (str): Pertanyaan dari pengguna
            
        Returns:
            str: Jawaban dari OpenAI
        """
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "Anda adalah asisten AI yang membantu menjawab pertanyaan pengguna dengan ramah dan informatif. Berikan jawaban singkat dan padat."},
                    {"role": "user", "content": query}
                ],
                max_tokens=150,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error when calling OpenAI API: {e}")
            return "Maaf, saya tidak dapat menjawab pertanyaan Anda saat ini."
    
    def get_response(self, query):
        """
        Mendapatkan respons berdasarkan pertanyaan pengguna
        
        Args:
            query (str): Pertanyaan dari pengguna
            
        Returns:
            str: Jawaban dari FAQ atau OpenAI
        """
        # Periksa apakah query kosong atau terlalu pendek
        if not query or len(query.strip()) < 3:
            return "Mohon masukkan pertanyaan yang lebih lengkap"
            
        # Preprocess query dengan format yang diharapkan oleh e5
        preprocessed_query = f"query: {query}"
        
        # Encode query pengguna
        query_embedding = self.model.encode(preprocessed_query, convert_to_tensor=True)
        
        # Hitung skor kemiripan cosine dengan semua pertanyaan FAQ
        cosine_scores = util.cos_sim(query_embedding, self.faq_embeddings)[0]
        
        # Dapatkan indeks dengan skor tertinggi
        best_idx = torch.argmax(cosine_scores).item()
        best_score = cosine_scores[best_idx].item()
        
        print(f"Pertanyaan: {query}")
        print(f"Kemiripan tertinggi: {best_score:.4f} dengan '{self.faq_questions[best_idx]}'")
        
        # Implementasi validasi semantik tambahan
        query_words = set(query.lower().split())
        faq_words = set(self.faq_questions[best_idx].lower().split())
        
        # Hitung overlap kata-kata kunci
        common_words = query_words.intersection(faq_words)
        common_words = {word for word in common_words if len(word) > 3}  # Fokus pada kata-kata penting
        
        # Jika skor kemiripan di atas threshold DAN ada kata kunci yang sama, berikan jawaban dari FAQ
        if best_score >= self.threshold and (len(common_words) > 0 or best_score > 0.95):
            print("Menggunakan jawaban dari FAQ")
            return self.faq_answers[best_idx]
        else:
            # Jika tidak ada kemiripan yang cukup, gunakan OpenAI untuk menjawab
            print("Tidak ada kemiripan yang cukup, menggunakan OpenAI")
            return self.get_openai_response(query)

# Inisialisasi Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS untuk semua domain

# Inisialisasi chatbot
chatbot = HybridFAQChatbot(threshold=0.85)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    user_message = data.get('message', '')
    
    if not user_message:
        return jsonify({'error': 'Message is required'}), 400
    
    response = chatbot.get_response(user_message)
    return jsonify({'response': response})


if __name__ == '__main__':
    print("Memulai API chatbot pada http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
