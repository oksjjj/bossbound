# BOSSBOUND — Python 강화학습 프로젝트

사람 플레이, 화면 없는 학습, AI 관전이 하나의 Python 게임 규칙을 공유합니다. M4 MacBook Pro에서 PyTorch MPS를 선택할 수 있습니다. 아직 학습 성능을 보장하는 완성된 AI는 아닙니다.

## 맥북 설치

압축을 풀고 터미널에서 프로젝트 폴더로 이동합니다. Python 3.12를 권장합니다. Apple Silicon용 arm64 Python을 사용하세요.

```bash
cd ~/Downloads/bossbound-python
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python check_device.py
```

`check_device.py`에서 `MPS available: True`를 확인하세요. False라면 Python 아키텍처, macOS 및 PyTorch 설치 상태를 확인하거나 `--device cpu`를 사용하세요. `auto`는 MPS → CUDA → CPU 순서로 선택합니다. 작은 신경망은 전송 비용 때문에 CPU가 더 빠를 수도 있으므로 실제 시간으로 비교하세요. MPS가 빠르다고 가정하지 않습니다.

새 터미널을 열면 다시 `cd`와 `source .venv/bin/activate`를 실행해야 합니다.

## 1. 사람이 플레이

```bash
python play.py
```

마우스로 사냥터·전투 행동·능력치 투자를 선택합니다. 공격을 선택하면 내 공격과 적의 반격이 순서대로 표시되며, 그 사이에 방어를 선택할 수 없습니다. 능력치 투자에는 턴이 들지 않습니다. 화면 오른쪽 아래에서 기록을 저장하고, R로 새 게임을 시작합니다. Esc는 종료입니다.

## 2. 규칙 기반 AI 관전

```bash
python watch.py --baseline weapon
```

처음에는 일시정지 상태입니다. 재생 또는 Space로 시작하고, N으로 한 행동씩 진행합니다. 1/2/4/8배속을 지원합니다. 공격·방어·획득·레벨업 메시지, 체력과 장비 진행도를 표시합니다.

사용 가능한 전략: `random`, `attack`, `mixed`, `defense`, `weapon`, `armor`, `potion`, `weapon-potion`, `both`, `both-potion`.

이 전략은 수작업 규칙입니다. 학습된 AI와 구분해서 평가하세요. 보스전에서는 생존 가능한 최단 행동 순서를 BFS로 탐색하므로, 기본 비교 대상으로 강한 편입니다.

## 3. PyTorch DQN 학습

```bash
python train.py --episodes 20000 --device mps --output runs/dqn
```

기본값은 규칙 기반 시연 없이 epsilon-greedy로 탐험하는 순수 DQN 학습입니다. 화면과 애니메이션은 실행하지 않습니다. 경험 재생, 타깃 네트워크, Huber loss, Adam, gradient clipping, 합법 행동 마스크를 사용합니다.

게임은 성공 보상이 늦게 나오는 환경입니다. 짧은 학습으로 보스 격파가 보장되지 않으며, 보스를 만나기 전 실패가 반복될 수 있습니다. 학습 로그에서 실제 성능을 확인하세요.

규칙 기반 시연을 이용하는 별도 실험:

```bash
python train.py --episodes 20000 --warmup 1000 --device mps --output runs/dqn-assisted
```

첫 1000판은 규칙 기반 전략으로 경험을 수집하고, 이후 DQN 탐험으로 전환합니다. 이 방법은 순수 RL과 다른 실험 조건입니다. 시연 구간의 성공률을 학습된 AI의 성능으로 보고하면 안 됩니다. CSV의 `policy_source` 열로 구분합니다.

출력:

- `config.json`: 실험 설정
- `metrics.csv`: 판별 성공·턴·점수·epsilon·loss·경험 수집 방식
- `latest.pt`: 100판마다 저장하는 모델
- `final.pt`: 마지막 모델

체크포인트는 추론용 가중치입니다. optimizer/replay/RNG 상태를 포함하는 완전한 학습 재개 파일은 아닙니다. 출력 폴더를 다르게 지정하면 실험을 보존할 수 있습니다.

## 4. 학습된 AI 관전 및 학습 중 모니터링

다른 터미널에서 가상환경을 활성화한 뒤:

```bash
python watch.py --model runs/dqn/latest.pt --device mps
```

100판 이상 학습해 파일이 생긴 후 실행합니다. 관전은 탐험 없이 Q값이 가장 큰 합법 행동을 선택합니다. R을 누르면 최신 모델을 다시 불러와 새 게임을 시작합니다. 학습과 관전을 동시에 실행할 수 있지만 GPU 자원을 공유하므로 학습이 느려질 수 있습니다. 관전을 CPU로 실행해도 됩니다.

마지막 모델을 관전하려면 `latest.pt` 대신 `final.pt`를 지정합니다.

## 5. 전략 비교

```bash
python evaluate.py --episodes 100 --output runs/baselines.csv
python evaluate.py --episodes 100 --model runs/dqn/final.pt --device mps --output runs/dqn-evaluation.csv
```

성공률, 성공한 판의 평균 클리어 턴, 실패 점수까지 포함한 평균 점수를 CSV로 저장합니다. 확정 지급 규칙은 현재 결정론적입니다. 같은 정책이면 다른 환경 시드에서도 같은 결과가 나옵니다. 무작위 정책의 행동 탐험은 시드에 따라 달라집니다. 환경 시드 반복만으로 다양한 맵에서 일반화했다고 주장하지 마세요. 학습 시드를 바꾼 별도 실행으로 안정성을 비교하세요.

규칙 기반 검증 결과: 공격 성장 46턴, 혼합 성장 51턴, 방어 성장 70턴, 무기 45턴, 방어구 47턴, 회복약 45턴. 규칙 기반 준비 전략 내부 비교이며 전체 행동 공간의 최적값 증명은 아닙니다.

## 6. 기록 재생

```bash
python watch.py --replay runs/replay-파일번호.json
```

UI에서 저장한 JSON의 행동을 같은 환경에 다시 입력합니다. 현재 확정 규칙에서는 동일 상태를 재현합니다. 게임 규칙이 바뀌면 예전 기록의 결과도 달라질 수 있습니다.

## 구조

```text
game/core.py       게임 규칙; 화면·학습 라이브러리와 독립
game/env.py        Gymnasium adapter, 상태 벡터와 행동 마스크
game/renderer.py   Pygame 화면·애니메이션·관전 제어
agents/baseline.py 규칙 기반 전략과 보스전 탐색
agents/dqn.py      PyTorch DQN, MPS/CPU/CUDA, 모델 저장
play.py            사람 플레이
train.py           학습
watch.py           AI 관전·기록 재생
evaluate.py        전략별 평가
check_device.py    설치·MPS 진단
tests/             핵심 규칙과 환경·모델 검사
assets/            한국어 글꼴 및 라이선스
```

## 상태·행동·보상

관측은 체력·공격·방어·레벨·경험치·투자 포인트·남은 턴·장비·회복약·획득 진행도·전투 종류·적 정보·보스 공격 단계로 구성한 26차원 float32 벡터입니다. 행동 12개는 `game/core.py`의 ACTIONS 순서로 고정되어 있습니다. `info['action_mask']` 및 `env.action_masks()`는 지금 합법적인 행동을 제공합니다. DQN은 선택과 bootstrap 양쪽 모두 마스크를 적용합니다.

장비: 폐광 6회 처치 → 공격 +4, 폐성채 6회 처치 → 방어 +2. 각각 1회만 획득하며 즉시 적용합니다. 약초 숲 3회 처치마다 회복약 1개, 시작 0개, 보유 제한 없음. 레벨업 공격 +2 / 방어 +1. 훈련장 HP 16·공격 3·XP 4, 폐광/폐성채 HP 18·공격 5·XP 1, 약초 숲 HP 14·공격 4·XP 1.

게임 점수: 성공 `100 + 2 × 남은 턴`, 실패 -30. 환경 보상은 이 값을 100으로 나눕니다. 다른 행동 보상은 0, gamma=1.0으로 빠른 성공이라는 실제 점수 목표를 유지합니다. 중간 몬스터·장비 보상은 추가하지 않았습니다.

90턴은 게임 규칙 안의 종료 조건이고 남은 턴이 관측에 있으므로 시간 초과도 `terminated=True`입니다. 외부 실행 중단과 구별됩니다. Gym adapter는 금지 행동을 상태 변경 없는 0보상으로 반환하고 `info['invalid']=True`를 표시합니다. 제공하는 AI는 마스크로 금지 행동을 피합니다. core는 잘못된 행동에 ValueError를 발생시킵니다.

## 검증 및 한계

```bash
python -m unittest discover -s tests -v
```

핵심 전투·획득·레벨업·종료 검사, Gymnasium checker, CPU DQN 업데이트와 모델 저장/로드, Pygame 화면 렌더링을 검증했습니다. Python과 기존 브라우저 엔진의 359개 상태 전이가 일치했습니다. M4/macOS/MPS에서 직접 실행한 검증은 아닙니다.

`runs/verification/`에는 짧은 CPU 학습 결과를 담았습니다. 800판 실험의 마지막 greedy DQN은 보스를 깨지 못했습니다. 실행 확인용이며 좋은 모델로 제시하지 않습니다.

## 참고

- PyTorch MPS: https://docs.pytorch.org/docs/stable/notes/mps.html
- Gymnasium: https://gymnasium.farama.org/api/env/
- Pygame: https://www.pygame.org/docs/
- 한국어 글꼴: Noto Sans KR, SIL Open Font License. assets/OFL.txt 포함.
