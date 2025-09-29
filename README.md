# 🚀 google-erp-app 팀원 실행 매뉴얼

## 1. 프로젝트 가져오기

```bash
git clone https://github.com/Jaeho777/google-erp-app.git
cd google-erp-app
```

---

## 2. 가상환경 만들기 + 실행

맥/리눅스:

```bash
python3 -m venv venv
source venv/bin/activate
```

윈도우:

```bash
python -m venv venv
venv\Scripts\activate
```

---

## 3. 패키지 설치

```bash
pip install -r requirements.txt
```

---

## 4. 서비스 계정 키 넣기

1. **`keys/service-account.json`** 파일을 받아서,
   `google-erp-app/keys/` 폴더 안에 넣어주세요.
   (이건 GitHub에 없어요 → 따로 공유됨)

2. 환경 변수 등록
   맥/리눅스:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="~/google-erp-app/keys/service-account.json"
```

윈도우(cmd):

```bash
set GOOGLE_APPLICATION_CREDENTIALS=google-erp-app\keys\service-account.json
```

---

## 5. 실행

```bash
streamlit run app.py --server.port=8080
```

실행하면 터미널에 뜨는 URL(보통 `http://localhost:8080`)을 브라우저에서 열면 됩니다.

---

✅ 요약:
`git pull → pip install -r requirements.txt → 키 파일 넣기 → 실행`
→ 바로 대시보드 확인 가능!

---

