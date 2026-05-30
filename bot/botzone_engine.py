import sys


def run_botzone_loop(agent_cls, decide_fn, with_debug_info=False):
    """Run Botzone Mahjong stdin/stdout protocol loop.

    decide_fn signature:
      - when with_debug_info=True: decide_fn(None, obs) -> (response: str, debug_info)
      - when with_debug_info=False: decide_fn(None, obs) -> response: str
    """
    input()  # startup handshake: "1"

    seatWind = None
    agent = None
    zimo = False
    angang = None

    while True:
        request = input()
        while not request.strip():
            request = input()
        request = request.split()

        if request[0] == "0":
            seatWind = int(request[1])
            agent = agent_cls(seatWind)
            agent.request2obs("Wind %s" % request[2])
            print("PASS")

        elif request[0] == "1":
            agent.request2obs(" ".join(["Deal", *request[5:]]))
            print("PASS")

        elif request[0] == "2":
            obs = agent.request2obs("Draw %s" % request[1])

            if with_debug_info:
                response, debug_info = decide_fn(None, obs)
            else:
                response = decide_fn(None, obs)
                debug_info = None

            response = response.split()
            if response[0] == "Hu":
                print("HU")
            elif response[0] == "Play":
                if with_debug_info:
                    print("PLAY {}\n{}".format(response[1], debug_info))
                else:
                    print("PLAY %s" % response[1])
            elif response[0] == "Gang":
                if with_debug_info:
                    print("GANG {}\n{}".format(response[1], debug_info))
                else:
                    print("GANG %s" % response[1])
                angang = response[1]
            elif response[0] == "BuGang":
                if with_debug_info:
                    print("BUGANG {}\n{}".format(response[1], debug_info))
                else:
                    print("BUGANG %s" % response[1])

        elif request[0] == "3":
            p = int(request[1])
            if request[2] == "DRAW":
                agent.request2obs("Player %d Draw" % p)
                zimo = True
                print("PASS")

            elif request[2] == "GANG":
                if p == seatWind and angang:
                    agent.request2obs("Player %d AnGang %s" % (p, angang))
                elif zimo:
                    agent.request2obs("Player %d AnGang" % p)
                else:
                    agent.request2obs("Player %d Gang" % p)
                print("PASS")

            elif request[2] == "BUGANG":
                obs = agent.request2obs("Player %d BuGang %s" % (p, request[3]))
                if p == seatWind:
                    print("PASS")
                else:
                    if with_debug_info:
                        response, _ = decide_fn(None, obs)
                    else:
                        response = decide_fn(None, obs)
                    if response == "Hu":
                        print("HU")
                    else:
                        print("PASS")

            else:
                zimo = False
                if request[2] == "CHI":
                    agent.request2obs("Player %d Chi %s" % (p, request[3]))
                elif request[2] == "PENG":
                    agent.request2obs("Player %d Peng" % p)

                obs = agent.request2obs("Player %d Play %s" % (p, request[-1]))
                if p == seatWind:
                    print("PASS")
                else:
                    if with_debug_info:
                        response, debug_info = decide_fn(None, obs)
                    else:
                        response = decide_fn(None, obs)
                        debug_info = None

                    response = response.split()
                    if response[0] == "Hu":
                        print("HU")
                    elif response[0] == "Pass":
                        if with_debug_info and debug_info:
                            print("PASS" + "\n{}".format(debug_info))
                        else:
                            print("PASS")
                    elif response[0] == "Gang":
                        print("GANG")
                        angang = None
                    elif response[0] in ("Peng", "Chi"):
                        obs = agent.request2obs("Player %d " % seatWind + " ".join(response))

                        if with_debug_info:
                            response2, debug_info2 = decide_fn(None, obs)
                            print(
                                " ".join(
                                    [
                                        response[0].upper(),
                                        *response[1:],
                                        response2.split()[-1],
                                    ]
                                )
                                + "\n{}".format(
                                    __import__("json").dumps(
                                        {"action": debug_info, "tile": debug_info2}
                                    )
                                )
                            )
                        else:
                            response2 = decide_fn(None, obs)
                            print(
                                " ".join(
                                    [
                                        response[0].upper(),
                                        *response[1:],
                                        response2.split()[-1],
                                    ]
                                )
                            )

                        agent.request2obs("Player %d Un" % seatWind + " ".join(response))

        print(">>>BOTZONE_REQUEST_KEEP_RUNNING<<<")
        sys.stdout.flush()
