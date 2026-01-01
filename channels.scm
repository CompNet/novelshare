(list (channel
        (name 'guix-aethor)
        (url "https://gitlab.com/Aethor/guix-aethor")
        (branch "main")
        (commit
         "87c1c77caf7a75388e9271bbcec1bdba6b35ccf7")
        (introduction
          (make-channel-introduction
            "b9b2fe220f3fedd8f22dcbd74ab6c817a62378e0"
            (openpgp-fingerprint
              "600B 943A D096 37ED B626  DDE8 2B43 8E4E BF57 55E9"))))
      (channel
        (name 'guix-science)
        (url "https://codeberg.org/guix-science/guix-science.git")
        (branch "master")
        (commit
          "b470b7cbecb01d4a8acd66137b988001e6b3cfd5")
        (introduction
          (make-channel-introduction
            "b1fe5aaff3ab48e798a4cce02f0212bc91f423dc"
            (openpgp-fingerprint
              "CA4F 8CF4 37D7 478F DA05  5FD4 4213 7701 1A37 8446"))))
      (channel
        (name 'guix-science-nonfree)
        (url "https://codeberg.org/guix-science/guix-science-nonfree.git")
        (branch "master")
        (commit
          "b94a466109e546b01eada60804661bdcd18e6199")
        (introduction
          (make-channel-introduction
            "58661b110325fd5d9b40e6f0177cc486a615817e"
            (openpgp-fingerprint
              "CA4F 8CF4 37D7 478F DA05  5FD4 4213 7701 1A37 8446"))))
      (channel
        (name 'guix)
        (url "https://git.guix.gnu.org/guix.git")
        (branch "master")
        (commit
          "4ce3182d41315ed8e7beb753934d7572d18b4cb6")
        (introduction
          (make-channel-introduction
            "9edb3f66fd807b096b48283debdcddccfea34bad"
            (openpgp-fingerprint
              "BBB0 2DDF 2CEA F6A8 0D1D  E643 A2A0 6DF2 A33A 54FA"))))
      (channel
        (name 'guix-past)
        (url "https://codeberg.org/guix-science/guix-past.git")
        (branch "master")
        (commit
          "473c942b509ab3ead35159d27dfbf2031a36cd4d")
        (introduction
          (make-channel-introduction
            "0c119db2ea86a389769f4d2b9c6f5c41c027e336"
            (openpgp-fingerprint
              "3CE4 6455 8A84 FDC6 9DB4  0CFB 090B 1199 3D9A EBB5")))))
