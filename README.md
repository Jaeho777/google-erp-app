# 📝 google-erp-app 실행 매뉴얼 (팀원용)

---

## 1. 프로젝트 가져오기

```bash
git clone https://github.com/Jaeho777/google-erp-app.git
cd google-erp-app
```

---

## 2. 가상환경 만들기 & 실행

### 맥/리눅스 🍎🐧

```bash
python3 -m venv venv
source venv/bin/activate
```

### 윈도우 💻

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

## 4. 서비스 계정 키 설정

1. **키 파일 받기**
   팀 리더(재호)가 공유하는 `service-account.json` 파일을 받습니다.

2. **폴더에 넣기**
   받은 파일을 이 경로에 넣어주세요:

   ```
   google-erp-app/keys/service-account.json
   ```

3. **환경 변수 등록**

   ### 맥/리눅스 🍎🐧

   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="~/google-erp-app/keys/service-account.json"
   ```

   ### 윈도우 💻 (cmd)

   ```bash
   set GOOGLE_APPLICATION_CREDENTIALS=google-erp-app\keys\service-account.json
   ```

   (⚠️ 이 명령어는 터미널을 껐다 켜면 다시 입력해야 합니다.)

---

## 5. 앱 실행

```bash
streamlit run app.py --server.port=8080
```

---

## 6. 접속

터미널에 이런 메시지가 뜹니다:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8080
```

👉 브라우저에서 `http://localhost:8080` 접속하면 대시보드 확인 가능합니다 ✅

---

## ❗ 자주 발생하는 오류 & 해결법

1. **`command not found: streamlit`**
   → `pip install streamlit` 다시 설치하세요.

2. **`DefaultCredentialsError` (인증 오류)**
   → `GOOGLE_APPLICATION_CREDENTIALS` 환경 변수를 다시 등록하세요.

3. **데이터가 안 나와요**
   → 키 파일이 최신 버전인지 확인하세요 (리더가 공유한 것만 사용).

---

✅ **팀원이 해야 할 건 딱 4가지**

1. GitHub에서 pull 받기
2. `pip install -r requirements.txt`
3. 키 파일 받아서 넣기
4. 실행하기

---

# 🔀 협업 매뉴얼 (브랜치 전략)

## 1. 메인 브랜치 확인

항상 `main` 브랜치에서 시작합니다.

```bash
git checkout main
git pull
```

---

## 2. 새 작업 브랜치 만들기

작업할 때는 무조건 새로운 브랜치에서 진행합니다.
브랜치 이름 규칙은 이렇게 맞춥시다:

* `이름/작업내용`
* 예시:

  * `jaeho/feature-dashboard`
  * `nohyunho/fix-bug-input`

```bash
git checkout -b 이름/작업내용
```

---

## 3. 작업 & 커밋

코드를 수정한 뒤,

```bash
git add .
git commit -m "대시보드 그래프 추가"
```

---

## 4. 브랜치 푸시

자신의 브랜치를 원격 저장소에 올립니다.

```bash
git push
오류 뜨면 오류에 적힌 git push- 어쩌구저쩌구 코드 복사 붙여넣기
```

---

## 5. Pull Request 생성

GitHub 웹사이트에서 **Pull Request(PR)**가 뜸.
이때 카톡방에 연락주기
---

## ❗ 주의할 점

* 절대 `main` 브랜치에서 직접 작업하지 말기 🚫
* 작업 시작 전 항상 `main` 최신화 (`git pull origin main`)
* 커밋 메시지는 간결하고 명확하게

---
