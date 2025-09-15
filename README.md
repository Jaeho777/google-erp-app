# 🌿 IMC GCP ERP 프로젝트 실행 매뉴얼 (팀원용, 로컬 무조건 실행 가능 버전)

---

## 1. 프로젝트 내려받기

터미널에서 차례대로 입력하세요:

```bash
cd ~/Desktop   # 바탕화면으로 이동
git clone https://github.com/Jaeho777/google-erp-app.git
cd google-erp-app
```

📂 바탕화면에 `google-erp-app` 폴더가 생깁니다.

---

## 2. 파이썬 설치 확인

```bash
python3 --version
```

👉 `Python 3.10 이상`이 나오면 OK.
안 나오면 [Python 공식 사이트](https://www.python.org/downloads/)에서 설치하세요.

---

## 3. 가상환경 만들기

```bash
python3 -m venv venv
source venv/bin/activate
```

* 성공하면 `(venv)` 표시가 터미널 맨 앞에 생깁니다.

---

## 4. 라이브러리 설치

```bash
pip install -r requirements.txt
```

👉 여기서 에러 나면 `pip install --upgrade pip` 먼저 하고 다시 실행하세요.

---

## 5. 서비스 계정 키 받기

1. 팀장(이재호)에게 `erp-key.json` 파일을 받습니다.
2. 이 파일을 꼭 여기다 넣으세요:

   ```
   google-erp-app/erp-key.json
   ```

📌 주의: 이 파일은 **비밀 열쇠**라서 깃허브에 절대 올리면 안 됩니다!

---

## 6. 환경 변수 설정

터미널에서:

```bash
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/Desktop/google-erp-app/erp-key.json"
```

👉 이건 매번 해야 합니다.
👉 귀찮으면 `~/.zshrc` 파일 맨 아래에 추가하면 자동 적용됩니다:

```bash
nano ~/.zshrc
# 맨 아래에 붙여넣기
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/Desktop/google-erp-app/erp-key.json"
```

저장 후:

```bash
source ~/.zshrc
```

---

## 7. 앱 실행하기

```bash
streamlit run app.py
```

* 자동으로 브라우저가 열립니다.
* 주소창에:

  ```
  http://localhost:8501
  ```

  입력하면 실행됩니다.

---

## 8. 사용할 수 있는 기능

1. **판매 입력 메뉴**

   * 날짜, 상품, 수량, 매출 입력 → BigQuery 테이블에 저장
2. **대시보드 메뉴**

   * 날짜별 매출 추이 그래프
   * 상품별 매출 비교 그래프
   * 데이터 테이블 보기

👉 즉, 로컬에서도 실제 ERP 대시보드 체험이 가능합니다!

---

## 9. 에러 해결 가이드

* ❌ `ModuleNotFoundError` → `pip install -r requirements.txt` 다시 실행
* ❌ `403 Access Denied` → 팀장에게 "권한 주세요"라고 말하기
* ❌ `404 Not Found` → `app.py` 안에 테이블 이름이 맞는지 확인

---

## 10. 개발 참여 (선택)

👉 새로운 기능을 만들고 싶으면:

```bash
git checkout -b feature-내이름
# 코드 수정
git add .
git commit -m "내 작업 설명"
git push origin feature-내이름
```

👉 팀장이 검토 후 합칩니다.
