
import argparse
import os

import cv2


def main():
    parser = argparse.ArgumentParser(description="Frames uit een video halen")
    parser.add_argument("video", help="pad naar het videobestand")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="om de hoeveel seconden een frame opslaan (standaard 1.0)")
    parser.add_argument("--out", default="frames/raw",
                        help="map waar de frames in komen (standaard frames/raw)")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"Kan de video niet openen: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    totaal_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duur = totaal_frames / fps if fps else 0
    stap = max(1, int(round(fps * args.interval)))

    os.makedirs(args.out, exist_ok=True)
    naam = os.path.splitext(os.path.basename(args.video))[0]

    print(f"Video: {duur:.0f} seconden, {fps:.1f} frames per seconde")
    print(f"We bewaren elke {args.interval} seconde (elk {stap}e frame)")

    opgeslagen = 0
    nummer = 0
    while True:
        gelukt, frame = cap.read()
        if not gelukt:
            break
        if nummer % stap == 0:
            tijd = nummer / fps
            bestand = os.path.join(args.out, f"{naam}_t{tijd:07.2f}s.jpg")
            cv2.imwrite(bestand, frame)
            opgeslagen += 1
        nummer += 1

    cap.release()
    print(f"Klaar! {opgeslagen} frames opgeslagen in {args.out}")


if __name__ == "__main__":
    main()
