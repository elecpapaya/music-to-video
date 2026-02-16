Apple Music 스타일 생성기 (Python v12)

v3에서 실패한 이유
- ffmpeg의 -filter_complex @file 방식이 Windows에서 줄바꿈/CRLF/빈줄 등에 민감하게 파싱되어
  'No such filter: ''' 같은 에러가 날 수 있습니다.

v4 변경점
- 필터 그래프를 '한 줄'로 만들어 subprocess 인자 리스트로 직접 전달 (shell 미사용)
  -> 특수문자/줄바꿈 문제 거의 사라짐
- 그래프는 여전히 output/<title>_filter.txt 로 저장(디버깅/커스터마이즈용)

v5 변경점 (콘텐츠 품질 강화)
- 출력 고정 동작을 `high` 품질 프리셋으로 강화:
  - 인코딩 CRF 18, ffmpeg preset slow
  - 최대 bitrate 16M / bufsize 32M
  - 오디오 256k
  - 60fps 기본 출력
  - 스케일링 필터 품질 향상(lanczos), 불필요한 잡음 효과 제거
- `--quality` 옵션 추가: `standard`, `high`, `ultra`
  - `ultra` 사용 시 CRF 16, 20M/40M, 오디오 320k

v6 변경점 (UI 재설계)
- 배경 분리 전략을 `blur+dark gradient layers`로 재구성해 조각/블록감 감소
- 좌측 커버 + 우측 정보 레이어 레이아웃 재배치
- 타이틀/아티스트/웨이브/재생바의 상대 위치 정렬 강화
- 같은 소스 기반에서 `--style` 기본은 `youtube`로 유지

v7 변경점 (audio-visualizer-python 벤치마크 반영)
- `youtube` 스타일에 미러 스펙트럼 바(상/하 대칭) 적용
- 중앙 글로우 라인 + 진행바 강조로 재생감 강화
- 배경을 더 어둡고 균일한 스테이지 톤으로 조정해 깨짐/조각감 완화

v8 변경점 (요청 반영)
- 동그라미 형식 이퀄라이저(`avectorscope` polar)로 변경
- 커버 이미지를 자르지 않도록 배경 생성 경로를 `scale+pad` 기반으로 변경

실행
python make_player_apple.py --song "Midnight whispers.wav" --cover "Midnight whispers.jpg" --title "Midnight whispers" --artist "Susan"
python make_player_apple.py --song "Midnight whispers.wav" --cover "Midnight whispers.jpg" --title "Midnight whispers" --artist "Susan" --quality ultra
python make_player_apple.py --song "Midnight whispers.wav" --cover "Midnight whispers.jpg" --title "Midnight whispers" --artist "Susan" --style youtube --quality ultra
python make_player_apple.py --song "Midnight whispers.wav" --cover "Midnight whispers.jpg" --title "Midnight whispers" --artist "Susan" --genre "K-POP"
python make_player_apple.py --title "Midnight whispers" --artist "Susan" --genre "K-POP"

옵션
- `--style youtube` (기본): 유튜브 업로드용으로 큰 커버, 우측 정보 패널, 강화된 진행 바/웨이브를 사용.
- `--style classic`: 기존 화면 구성(단순한 오른쪽 패널형 레이아웃).
- `--title`만 넣으면 `<title>.wav/.mp3/...` 와 `<title>.jpg/.png/...` 를 자동 탐색해서 입력 사용
- `--genre`: 썸네일 배지 문구(예: `K-POP`, `R&B`)
- `--tagline`: 썸네일 보조 문구(선택)
- `--skip-thumbnail`: 썸네일 생성 생략

결과
- output\Midnight whispers.mp4
- output\Midnight whispers_youtube_thumb.jpg
- output\Midnight whispers_filter.txt
