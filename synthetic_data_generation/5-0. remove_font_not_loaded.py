import os
from fontTools.ttLib import TTFont
from multiprocessing import Pool, cpu_count

# Directory containing fonts
font_dir = '11954_handwritten_Fonts_all'

# Allowed font file extensions
allowed_extensions = ['.otf', '.ttf', '.OTF', '.TTF']

def is_valid_font(file_path):
    try:
        # Try loading the font with TTFont
        TTFont(file_path)
        return True  # Font successfully loaded
    except Exception:
        return False  # Failed to load the font

def process_file(filename):
    file_path = os.path.join(font_dir, filename)

    # Check if the file is a valid font by extension
    if not any(filename.endswith(ext) for ext in allowed_extensions):
        print(f"Removing non-font file: {filename}")
        os.remove(file_path)  # Remove files that don't have a valid extension
        return 'removed'
    else:
        # Check if it's a valid font by loading it
        if not is_valid_font(file_path):
            print(f"Removing invalid font file: {filename}")
            os.remove(file_path)  # Remove font files that failed to load
            return 'removed'
        else:
            return 'remaining'  # Font file is valid

if __name__ == '__main__':
    # Get list of files
    filenames = os.listdir(font_dir)

    # Create a pool of worker processes equal to the number of CPU cores
    pool = Pool(processes=cpu_count())

    # Process files in parallel
    results = pool.map(process_file, filenames)

    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()

    # Count the results
    removed_count = results.count('removed')
    remaining_count = results.count('remaining')

    # Print the results
    print(f"Total removed files: {removed_count}")
    print(f"Total remaining valid font files: {remaining_count}")

# Total removed files: 238 + 387
# Total remaining valid font files: 17365

#238 removed
# Removing invalid font file: ._Familior.otf
# Removing invalid font file: ._Leonetta-Script.otf
# Removing non-font file: Darkest Saturday Outline.woff
# Removing invalid font file: ._Bunch Blossoms Personal Use.ttf
# Removing non-font file: ._froggyprincess-regular.svg
# Removing non-font file: Senja Free Trial.woff
# Removing invalid font file: ._southpaw.ttf
# Removing non-font file: knewave-outline-webfont.eot
# Removing invalid font file: ._FlunkiesBB.otf
# Removing non-font file: knewave-webfont.eot
# Removing invalid font file: ._Rantliyer.ttf
# Removing invalid font file: ._Bella Fashion Personal Use.ttf
# Removing invalid font file: ._Feri Candi.otf
# Removing invalid font file: ._GelPenLight.ttf
# Removing non-font file: Senja Free Trial.woff2
# Removing invalid font file: ._HandTIMES.ttf
# Removing invalid font file: ._HeySweety.ttf
# Removing invalid font file: ._KomixCon-Bold Italic 2.ttf
# Removing invalid font file: ._Jerhiyof.ttf
# Removing non-font file: Amantha woff 2.woff2
# Removing invalid font file: ._HonuzimaRegular.ttf
# Removing invalid font file: ._Hunthers Dwayne.ttf
# Removing invalid font file: ._KINGDOMSTORIA.ttf
# Removing invalid font file: ._Reguloza-Regular.ttf
# Removing invalid font file: ._SporkBold.ttf
# Removing invalid font file: ._Reguloza-Regular.otf
# Removing invalid font file: ._Newrotic.ttf
# Removing invalid font file: ._Allise.ttf
# Removing invalid font file: ._legworkDEMO.otf
# Removing invalid font file: ._Hotham .ttf
# Removing invalid font file: ._SanyCimahen.ttf
# Removing invalid font file: ._Allegratta Personal Use.ttf
# Removing invalid font file: ._California.ttf
# Removing invalid font file: ._FUNCY KIDS!.ttf
# Removing invalid font file: ._Sellviny Queen.ttf
# Removing non-font file: ._Candelia.woff
# Removing invalid font file: ._Vires Gollem.ttf
# Removing invalid font file: ._Oyange-Brush.otf
# Removing invalid font file: ._Shaky Hand Some Comic_bold.otf
# Removing non-font file: ._Beleriand.woff
# Removing non-font file: fabfeltscript-bold.svg
# Removing invalid font file: ._TequilaSunset.otf
# Removing non-font file: SebladeBlackItalic.woff
# Removing invalid font file: ._Spork.ttf
# Removing invalid font file: ._SummerVacation.otf
# Removing invalid font file: ._Sugih Janji.ttf
# Removing invalid font file: ._BellisyaSignature.ttf
# Removing invalid font file: ._Kenangan-Regular.otf
# Removing non-font file: Sundae Bite Free Trial.woff
# Removing invalid font file: ._Engine-Regular.otf
# Removing invalid font file: ._BabyWorld.otf
# Removing non-font file: ._froggyprincess-regular.eot
# Removing invalid font file: ._Amelliz.ttf
# Removing invalid font file: ._Secretss Personal use.ttf
# Removing non-font file: Killing Me Free Trial.woff2
# Removing non-font file: Shamilove woff 1.woff
# Removing invalid font file: ._Oyange.ttf
# Removing non-font file: Show Up! Free Trial.woff2
# Removing invalid font file: ._Sweetheart_Script_free.ttf
# Removing non-font file: Haloha Free Trial.woff
# Removing invalid font file: ._CandyQelling.otf
# Removing non-font file: Robb-Regular.woff
# Removing invalid font file: ._news junkie DEMO.otf
# Removing invalid font file: ._FUNCY KIDS!.otf
# Removing non-font file: Snowy Holiday.woff
# Removing non-font file: Battgge.woff2
# Removing invalid font file: ._Anythings.otf
# Removing invalid font file: ._StopYelling-Regular.otf
# Removing invalid font file: ._blikfangDEMO.otf
# Removing invalid font file: ._LandscapePERSONALUSE.otf
# Removing invalid font file: ._Captain Jack demo.ttf
# Removing invalid font file: ._LoveWhisper-Personal Use.ttf
# Removing invalid font file: ._Jerhiyof.otf
# Removing invalid font file: ._Aye Matey demo.ttf
# Removing non-font file: knewave-outline-webfont.woff
# Removing invalid font file: ._Grandstander-clean.ttf
# Removing invalid font file: ._Gelathy.ttf
# Removing invalid font file: ._Sugarly.otf
# Removing invalid font file: ._Wenykidos.otf
# Removing invalid font file: ._Ratih Hyun.otf
# Removing invalid font file: ._PROPAGANDA SIGHT PERSONAL USE.ttf
# Removing invalid font file: ._SmudgeStickOblique.ttf
# Removing invalid font file: ._charmeladeDEMO.otf
# Removing non-font file: froggyprincess-regular.woff
# Removing invalid font file: ._Amaliyah-Regular.ttf
# Removing non-font file: Hello Mozza Outline.woff
# Removing invalid font file: ._Sugarly.ttf
# Removing invalid font file: ._raustila-Regular.ttf
# Removing invalid font file: ._FB-Strawberry.ttf
# Removing non-font file: Yellow Fontie.woff
# Removing invalid font file: ._Burning Heart Personal Use.ttf
# Removing invalid font file: ._BalhgiRizetons.otf
# Removing invalid font file: ._Honey Lips Personal Use.ttf
# Removing invalid font file: ._GelPen.ttf
# Removing invalid font file: ._Jelly Belty!.ttf
# Removing invalid font file: ._LindbergHand.otf
# Removing invalid font file: ._Atmospherica Personal Use.ttf
# Removing invalid font file: ._Great Day Personal Use.ttf
# Removing invalid font file: ._Gellato.ttf
# Removing invalid font file: ._GothicHandDirty_bold.ttf
# Removing invalid font file: ._SWQuickStaffMeeting.ttf
# Removing invalid font file: ._AmazingKids.otf
# Removing invalid font file: ._Bite Chocolate Personal Use.ttf
# Removing invalid font file: ._KomixCon.otf
# Removing non-font file: Ojosujono.woff
# Removing invalid font file: ._Rattu Aqilla.otf
# Removing non-font file: Arista Signature.woff2
# Removing non-font file: THOERTIEN.woff
# Removing invalid font file: ._Vires Gollem.otf
# Removing invalid font file: ._Bristol.otf
# Removing non-font file: Bluffton Free Trial.woff2
# Removing non-font file: Candelia.woff
# Removing invalid font file: ._GelPenUprightLightCondensed.ttf
# Removing invalid font file: ._PajamaPantsLightItalic.ttf
# Removing invalid font file: ._snubnose DEMO.otf
# Removing non-font file: GeopieMoorpie.woff
# Removing invalid font file: ._partyjamDEMO.otf
# Removing invalid font file: ._Run to the Hills Personal Use.ttf
# Removing invalid font file: ._Ambar Pearl Personal Use.ttf
# Removing invalid font file: ._Shanti Creny.otf
# Removing non-font file: Qanethya woff 1.woff
# Removing invalid font file: ._Silverline_Script_Demo.otf
# Removing non-font file: knewave-webfont.svg
# Removing non-font file: AGIS-Italic.woff
# Removing non-font file: MonalisaFont.woff
# Removing invalid font file: ._HVD_Bodedo.ttf
# Removing invalid font file: ._Claudia Personal Use.ttf
# Removing invalid font file: ._Janyss Brush.otf
# Removing non-font file: Arista Signature.woff
# Removing non-font file: Sunset Club Free Trial.woff
# Removing non-font file: SebladeBlackRegular2.woff
# Removing invalid font file: ._Oyange-Brush.ttf
# Removing non-font file: Raliangi WOFF.woff
# Removing invalid font file: ._spektakel DEMO.otf
# Removing non-font file: agathsya.woff
# Removing invalid font file: ._Body&Soul Personal Use.ttf
# Removing non-font file: Niagra Faults.woff
# Removing invalid font file: ._GelPenUpright.ttf
# Removing invalid font file: ._DecalkBoldItalic.ttf
# Removing invalid font file: ._GelPenSerifLight.ttf
# Removing invalid font file: ._AGATM___.TTF
# Removing invalid font file: ._SporkBoldItalic.ttf
# Removing non-font file: Tribista.woff
# Removing invalid font file: ._Allicia Personal Use.ttf
# Removing non-font file: TfFancy Free Trial.woff2
# Removing non-font file: SebladeBlackItalic2.woff
# Removing invalid font file: ._Mady Risaw.ttf
# Removing invalid font file: ._Sunday Morning Personal Use.otf
# Removing invalid font file: ._Aquilla.ttf
# Removing invalid font file: ._PEIXE___.ttf
# Removing non-font file: Weathertop.woff
# Removing invalid font file: ._Zerocalcare-Script-Bold-trial.ttf
# Removing non-font file: Amulman-Bold.svg
# Removing invalid font file: ._Bimbo-Whiteboard-trial.ttf
# Removing non-font file: Haloha Free Trial.woff2
# Removing non-font file: Darkest Saturday Aged.woff2
# Removing invalid font file: ._PajamaPants.ttf
# Removing non-font file: Qanethya woff 2.woff
# Removing invalid font file: ._Claudina Personal Use.ttf
# Removing invalid font file: ._Elowen.ttf
# Removing invalid font file: ._Carpenters Personal Use.ttf
# Removing non-font file: Hello Mozza.woff
# Removing non-font file: ._froggyprincess-regular.woff2
# Removing non-font file: SebladeBlackOutline1RR.woff
# Removing invalid font file: ._Shaky Hand Some Comic_3D.otf
# Removing invalid font file: ._SWReallyQuickStaffMeeting.ttf
# Removing non-font file: Ebony-Regular.woff
# Removing invalid font file: ._Beth-Ellen-2.0.otf
# Removing invalid font file: ._Rathury.ttf
# Removing invalid font file: ._RAYJOHNS.ttf
# Removing invalid font file: ._Tajamuka Script.ttf
# Removing invalid font file: ._nikotinusDEMO.otf
# Removing non-font file: Helsinky Free Trial.woff
# Removing invalid font file: ._Theodora Personal Use.ttf
# Removing invalid font file: ._Mady Risaw.otf
# Removing invalid font file: ._KaoriGelBold.ttf
# Removing invalid font file: ._Ajuslly.otf
# Removing non-font file: Rouweth.woff
# Removing invalid font file: ._Music Magic Personal Use.ttf
# Removing non-font file: agathsya.eot
# Removing invalid font file: ._Daisuky Fancy.ttf
# Removing invalid font file: ._LandscapeLandPERSONALUSE.otf
# Removing invalid font file: ._Hotel Costes Personal Use.ttf
# Removing non-font file: Ygritte-Regular.woff
# Removing invalid font file: ._SleeplessonPlus-Regular.otf
# Removing invalid font file: ._Hey Kidos!.otf
# Removing invalid font file: ._Elegant hand Script.otf
# Removing invalid font file: ._FabfeltScript-Bold.otf
# Removing invalid font file: ._TT Rabbits Elf DEMO.otf
# Removing invalid font file: ._Honuzima.otf
# Removing non-font file: Sugarly.woff
# Removing invalid font file: ._Anythings.ttf
# Removing invalid font file: ._CandyQelling.ttf
# Removing invalid font file: ._Rantliyer.otf
# Removing invalid font file: ._Westbury-Signature-Demo-Version.otf
# Removing invalid font file: ._SanyCimahen.otf
# Removing invalid font file: ._Gotten.ttf
# Removing invalid font file: ._PajamaPantsItalic.ttf
# Removing invalid font file: ._Amelliz.otf
# Removing invalid font file: ._Wall Paper Personal Use.ttf
# Removing invalid font file: ._Orange Personal Use.ttf
# Removing invalid font file: ._foolishpeopleDEMO.otf
# Removing invalid font file: ._To The Point.ttf
# Removing invalid font file: ._PrinsesstartaMediumItalicDEMO.ttf
# Removing invalid font file: ._KomixCon-Bold.otf
# Removing invalid font file: ._SnowHut.ttf
# Removing invalid font file: ._SivarPro.otf
# Removing invalid font file: ._Manthesy.otf
# Removing invalid font file: ._Curely-Free Typeface.otf
# Removing invalid font file: ._GelPenUprightCondensed.ttf
# Removing non-font file: RainaHusnaRegular.woff
# Removing invalid font file: ._BabyWorld.ttf
# Removing non-font file: quirky_nots.woff
# Removing invalid font file: ._your flames DEMO.otf
# Removing invalid font file: ._Amarula Personal Use.ttf
# Removing invalid font file: ._BellisyaSignature.otf
# Removing invalid font file: ._Sabina Angellica.otf
# Removing invalid font file: ._Brilhant PERSONAL USE.ttf
# Removing invalid font file: ._FB-MommaHero1.ttf
# Removing invalid font file: ._VigtigperDEMO.otf
# Removing invalid font file: ._NaturalSignature.otf
# Removing non-font file: DevonSweet.woff
# Removing non-font file: armed-webfont.svg
# Removing invalid font file: ._Berton-Voyage-trial.ttf
# Removing non-font file: SnoorksRegular.woff
# Removing invalid font file: ._NEWYORK.ttf
# Removing invalid font file: ._YeahPapa.ttf
# Removing invalid font file: ._KomixCon-Bold Italic.otf
# Removing invalid font file: ._Candy Yefumy.ttf
# Removing invalid font file: ._Ajuslly.ttf
# Removing invalid font file: ._Caranda Personal Use.ttf
# Removing non-font file: LeagueScriptNumberOne-webfont.woff
# Removing invalid font file: ._Grandstander-clean.otf
# Removing non-font file: quirky_nots.eot
# Removing invalid font file: ._The_OG.ttf
# Removing invalid font file: ._Antonine Personal Use.ttf
# Removing non-font file: Amulman-Light.svg
# Removing invalid font file: ._PrinsesstartaBoldDEMO.ttf
# Removing invalid font file: ._ToastedCinnamon.otf
# Removing non-font file: AGIS.woff
# Removing non-font file: Lovebird.woff
# Removing invalid font file: ._ComicDylans.otf
# Removing non-font file: Sunset Club Free Trial.woff2
# Removing invalid font file: ._GelPenUprightHeavyCondensed.ttf
# Removing non-font file: Koplok Free Trial.woff
# Removing invalid font file: ._HONZO.otf
# Removing non-font file: Sarttink Signature.woff
# Removing invalid font file: ._PajamaPantsBold.ttf
# Removing invalid font file: ._AgendaKing.otf
# Removing invalid font file: ._Bagus Stanlley.ttf
# Removing invalid font file: ._ComicDylans.ttf
# Removing invalid font file: ._KomixCon 2.ttf
# Removing invalid font file: ._Balgeris.ttf
# Removing invalid font file: ._GothicHandDirty.ttf
# Removing invalid font file: ._Ames-Regular.otf
# Removing non-font file: Shockwave.woff
# Removing non-font file: Koplok Free Trial.woff2
# Removing non-font file: Battilla.woff
# Removing invalid font file: ._PrettyRosse.ttf
# Removing invalid font file: ._Romantisk DEMO.otf
# Removing non-font file: My Home Free Trial.woff
# Removing invalid font file: ._Sabina Angellica.ttf
# Removing non-font file: SebladeBlackOutline2R.woff
# Removing non-font file: Helsinky Free Trial.woff2
# Removing invalid font file: ._Giraffenhals.otf
# Removing non-font file: Baby Crab Italic.woff
# Removing invalid font file: ._Freehand-Blockletter-Bold-trial.ttf
# Removing invalid font file: ._Jarida.ttf
# Removing non-font file: Fronzy Free Trial.woff2
# Removing invalid font file: ._bottle party DEMO.otf
# Removing invalid font file: ._Engine-Italic.otf
# Removing non-font file: Ameyallinda Signature.woff
# Removing invalid font file: ._Clint Marker.ttf
# Removing non-font file: Tibet.woff
# Removing invalid font file: ._Aisbum Slashey.ttf
# Removing non-font file: Homework.woff
# Removing invalid font file: ._Kangtoni.otf
# Removing invalid font file: ._Giraffenhals_bold.otf
# Removing non-font file: Coffee Written Bold.woff
# Removing invalid font file: ._Bimbo-Jumbo-trial.ttf
# Removing invalid font file: ._Shanti Creny.ttf
# Removing invalid font file: ._Biund.otf
# Removing non-font file: Sketch 3D.woff
# Removing non-font file: The Barethos.woff
# Removing invalid font file: ._Girdens.ttf
# Removing non-font file: My Kids Handwritten-Basic.woff2
# Removing non-font file: southpawwebfont.woff
# Removing non-font file: RainaHusnaItalic.woff2
# Removing invalid font file: ._froggyprincess-regular.ttf
# Removing non-font file: Clegane-Regular.woff
# Removing non-font file: Ragellia Mellinda.woff
# Removing invalid font file: ._KomixCon Italic.ttf
# Removing invalid font file: ._Thyme.ttf
# Removing non-font file: Austin.woff
# Removing invalid font file: ._handwritten-pittorifamosi-02.ttf
# Removing invalid font file: ._Casual Chance Personal Use.ttf
# Removing invalid font file: ._Desert Queen Personal Use.ttf
# Removing non-font file: Ameyallinda Signature.svg
# Removing non-font file: deSolidia.woff
# Removing non-font file: Fronzy Free Trial.woff
# Removing non-font file: fabfeltscript-bold.woff
# Removing invalid font file: ._Beredith Personal use.ttf
# Removing non-font file: ._Butterskull.woff
# Removing non-font file: Ojosujono.svg
# Removing invalid font file: ._Andorra Personal Use.ttf
# Removing invalid font file: ._KomixCon.ttf
# Removing invalid font file: ._Camellio-Regular.ttf
# Removing invalid font file: SillyGalDemo-pJpa.otf
# Removing invalid font file: ._Wenykidos.ttf
# Removing invalid font file: ._BalhgiRizetonsRegular.ttf
# Removing non-font file: medicall.woff
# Removing invalid font file: ._KomixCon Italic 2.ttf
# Removing invalid font file: ._FB-Strawberry.otf
# Removing invalid font file: ._raustila-Regular.otf
# Removing non-font file: Candyful - Free Trial.eot
# Removing non-font file: froggyprincess-regular.woff2
# Removing invalid font file: ._Sellviny Queen.otf
# Removing non-font file: Amantha woff 1.woff
# Removing non-font file: ._southpawwebfont.eot
# Removing invalid font file: ._Elowen.otf
# Removing invalid font file: ._Sleeplesson-Regular.ttf
# Removing invalid font file: ._KomixCon-Bold 2.ttf
# Removing non-font file: Gomgom Handwrite-Basic.woff2
# Removing invalid font file: ._PrinsesstartaLightDEMO.ttf
# Removing invalid font file: ._Blackway Brush.ttf
# Removing invalid font file: ._Allise.otf
# Removing invalid font file: ._Afternoon in Stereo Personal Use.ttf
# Removing non-font file: ._deSolidia.woff
# Removing invalid font file: ._dummkopfDEMO.otf
# Removing non-font file: Sketch Handwriting.woff
# Removing invalid font file: ._Elegant hand Script.ttf
# Removing invalid font file: ._Walytime.ttf
# Removing non-font file: ._BellisyaSignature.woff
# Removing non-font file: Esalina Julite.woff
# Removing invalid font file: ._SpringTime Personal Use.ttf
# Removing invalid font file: ._pandoramaDEMO.otf
# Removing non-font file: armed-webfont.eot
# Removing invalid font file: ._SummerVacation.ttf
# Removing non-font file: Baby Crab Italic.woff2
# Removing invalid font file: ._Janyss Brush.ttf
# Removing invalid font file: ._HandTIMES.otf
# Removing non-font file: Shamilove woff 2.woff
# Removing invalid font file: ._Feri Candi.ttf
# Removing invalid font file: ._Romello.ttf
# Removing invalid font file: ._Cameliya.otf
# Removing invalid font file: ._KaoriGel.ttf
# Removing invalid font file: ._PrinsesstartaLightItalicDEMO.ttf
# Removing invalid font file: ._California Sun Personal Use.ttf
# Removing non-font file: My Home Free Trial.woff2
# Removing invalid font file: ._Cameliya.ttf
# Removing invalid font file: ._Girdens.otf
# Removing non-font file: Parsley.woff
# Removing invalid font file: ._Remnant-Regular.ttf
# Removing invalid font file: ._madpakkeDEMO.otf
# Removing invalid font file: ._Ananda Black Personal Use.ttf
# Removing non-font file: ._southpawwebfont.woff
# Removing invalid font file: ._Candice Personal Use.ttf
# Removing invalid font file: ._Jelly Belty!.otf
# Removing non-font file: Battgge.woff
# Removing invalid font file: ._deSolidia.ttf
# Removing invalid font file: ._SmudgeStick.ttf
# Removing invalid font file: ._FB-MommaHero1.otf
# Removing invalid font file: ._dirtyDeoHandInk.ttf
# Removing invalid font file: ._Tylerwolf.otf
# Removing invalid font file: ._Butterskull.ttf
# Removing invalid font file: ._MARKMF__.TTF
# Removing invalid font file: ._Camellia.otf
# Removing invalid font file: ._Revij Anovik.ttf
# Removing invalid font file: ._Candelia.otf
# Removing non-font file: Darkest Saturday Rough.woff
# Removing invalid font file: ._PrettyRosseSwash.ttf
# Removing invalid font file: ._Valentine Day Personal Use.ttf
# Removing invalid font file: ._RomantikaHidupRegular.ttf
# Removing invalid font file: ._DirlyBelly-Regular.otf
# Removing invalid font file: ._LuxuriousDEMO.otf
# Removing non-font file: Girly Minnie.woff
# Removing non-font file: knewave-webfont.woff
# Removing invalid font file: ._Beleriand.ttf
# Removing invalid font file: ._hyggebukserDEMO.otf
# Removing invalid font file: ._Slimamif.ttf
# Removing non-font file: Baby Crab.woff2
# Removing invalid font file: ._Hells Kittchen Devil God_bold.otf
# Removing invalid font file: ._Childrens Party Personal Use.ttf
# Removing invalid font file: ._JackdawsDEMO.otf
# Removing invalid font file: ._Candy Yefumy.otf
# Removing non-font file: fabfeltscript-bold.eot
# Removing non-font file: The Barethos.woff2
# Removing invalid font file: ._Lovely Crafter Demo Version.otf
# Removing invalid font file: ._FroggyPrincess-Regular.otf
# Removing invalid font file: ._FB-MommaHero2.ttf
# Removing invalid font file: ._Kangtoni.ttf
# Removing non-font file: KanyaRegular.woff
# Removing invalid font file: ._BLOWUP PERSONAL USE.ttf
# Removing invalid font file: ._Revij Anovik.otf
# Removing non-font file: ._southpawwebfont.svg
# Removing non-font file: Scarematehy Free Trial.woff
# Removing invalid font file: ._FingerType.ttf
# Removing invalid font file: ._lemonismDEMO.otf
# Removing invalid font file: ._GelPenUprightHeavy.ttf
# Removing non-font file: Scarematehy Free Trial.woff2
# Removing invalid font file: ._Zerocalcare-Script-trial.ttf
# Removing non-font file: Sketch Script Cool.woff
# Removing invalid font file: ._Bathi.otf
# Removing invalid font file: ._NACHOTL_.ttf
# Removing invalid font file: ._Blackberry Jam Personal Use.ttf
# Removing invalid font file: ._PrinsesstartaMediumDEMO.ttf
# Removing invalid font file: ._SeindahCinttya.otf
# Removing non-font file: SebladeBlackOutline1R.woff
# Removing invalid font file: ._finurlig DEMO.otf
# Removing non-font file: TfFancy Free Trial.woff
# Removing non-font file: Bellania.woff
# Removing invalid font file: ._Hey Kidos!.ttf
# Removing non-font file: Bellania.svg
# Removing invalid font file: ._Sang Dewi.ttf
# Removing invalid font file: ._Decalk.ttf
# Removing non-font file: LaNantes.woff
# Removing invalid font file: ._AmazingKids.ttf
# Removing invalid font file: ._NaturalSignature.ttf
# Removing invalid font file: ._PROPAGANDA SIGHT SHADOW PERSONAL USE.ttf
# Removing non-font file: LeagueScriptNumberOne-webfont.eot
# Removing non-font file: RainaHusnaItalic.woff
# Removing invalid font file: ._NoteToSelf-Regular.otf
# Removing non-font file: knewave-outline-webfont.svg
# Removing invalid font file: ._Curely-Free Typeface.ttf
# Removing non-font file: armed-webfont.woff
# Removing non-font file: Darkest Saturday.woff2
# Removing non-font file: Coffee Written Italic.woff
# Removing non-font file: Coffee Written.woff
# Removing invalid font file: ._Giraffenhals_condensed.otf
# Removing invalid font file: ._ANGEL___.ttf
# Removing invalid font file: ._ButterzoneDEMO.otf
# Removing invalid font file: ._JANGKIDS.ttf
# Removing invalid font file: ._Sugih Janji.otf
# Removing invalid font file: ._gymnastik DEMO.otf
# Removing invalid font file: ._Gellato.otf
# Removing non-font file: Homework.svg
# Removing invalid font file: ._HVD_Edding.otf
# Removing invalid font file: ._Bagus Stanlley.otf
# Removing invalid font file: ._Bellamy Signature.otf
# Removing non-font file: RainaHusnaRegular.woff2
# Removing non-font file: Valentine Moon woff 2.woff
# Removing invalid font file: ._Dragging-Canoe.otf
# Removing invalid font file: ._ChewedPenBB_ital.otf
# Removing invalid font file: ._Aperly.ttf
# Removing invalid font file: ._fabfeltscript-bold.ttf
# Removing non-font file: Baleno Handi.woff
# Removing invalid font file: ._Pagi Senja.otf
# Removing non-font file: Arya-Regular.woff
# Removing invalid font file: ._KomixCon-Bold.ttf
# Removing invalid font file: ._Pagi Senja.ttf
# Removing invalid font file: ._Freeride.otf
# Removing invalid font file: ._Rattu Aqilla.ttf
# Removing invalid font file: ._PajamaPantsBoldItalic.ttf
# Removing non-font file: SebladeBlackOutline2RR.woff
# Removing non-font file: Bluffton Free Trial.woff
# Removing non-font file: My Kids Handwritten-Basic.woff
# Removing invalid font file: ._Bimbo-Regular-trial.ttf
# Removing non-font file: froggyprincess-regular.eot
# Removing invalid font file: ._Hells Kittchen Devil God.ttf
# Removing invalid font file: ._Ananda Personal Use.ttf
# Removing invalid font file: ._JANGKIDS.otf
# Removing non-font file: Darkest Saturday.woff
# Removing invalid font file: ._Altavista Personal Use.ttf
# Removing invalid font file: ._Tukiyem.ttf
# Removing invalid font file: ._Sleeplesson-Regular.otf
# Removing invalid font file: ._Bimbo-Dripping-Jumbo-trial.ttf
# Removing invalid font file: ._Billy Argel Font___.ttf
# Removing non-font file: Darkest Saturday Aged.woff
# Removing invalid font file: ._Oyange.otf
# Removing invalid font file: ._PoetsenOne-Regular.otf
# Removing invalid font file: ._Tukiyem.otf
# Removing non-font file: froggyprincess-regular.svg
# Removing non-font file: ._fabfeltscript-bold.woff2
# Removing non-font file: Bellmetta.svg
# Removing invalid font file: ._southpaw.otf
# Removing non-font file: Darkest Saturday Rough.woff2
# Removing invalid font file: ._Sang Dewi.otf
# Removing invalid font file: ._Adalgisa Personal Use.ttf
# Removing invalid font file: ._DecalkItalic.ttf
# Removing non-font file: SebladeBlackRegular.woff
# Removing non-font file: Bellmetta.woff
# Removing invalid font file: ._Aisbum Slashey.otf
# Removing invalid font file: ._Jacyking.otf
# Removing non-font file: Valentine Moon woff 1.woff
# Removing invalid font file: ._Light And Airy.ttf
# Removing invalid font file: ._Butterskull.otf
# Removing invalid font file: ._Leonetta-Serif.otf
# Removing invalid font file: ._Romochka.otf
# Removing non-font file: From Street Art.woff
# Removing non-font file: ._fabfeltscript-bold.woff
# Removing invalid font file: ._Aquilla.otf
# Removing non-font file: Everything Calligraphy - WOFF.woff
# Removing invalid font file: ._Halgonak.ttf
# Removing invalid font file: ._Anastasia Script Personal Use.ttf
# Removing invalid font file: ._BLOW ME PERSONAL USE.ttf
# Removing invalid font file: ._Got to be Real Personal Use.ttf
# Removing invalid font file: ._Manthesy.ttf
# Removing non-font file: Show Up! Free Trial.woff
# Removing non-font file: Miss Nelly.woff
# Removing non-font file: southpawwebfont.eot
# Removing invalid font file: ._kidsrock DEMO.otf
# Removing invalid font file: ._CatCafe.ttf
# Removing invalid font file: ._observantDEMO.otf
# Removing invalid font file: ._SWStaffMeeting.ttf
# Removing invalid font file: ._KINGDOMSTORIA.otf
# Removing invalid font file: ._Bimbo-Finetip-trial.ttf
# Removing invalid font file: ._southpawwebfont.ttf
# Removing non-font file: Killing Me Free Trial.woff
# Removing non-font file: Playkidz.woff
# Removing invalid font file: ._SporkItalic.ttf
# Removing invalid font file: ._Candelia.ttf
# Removing non-font file: Sketch Wall.woff
# Removing invalid font file: ._ChewedPenBB.otf
# Removing non-font file: Butterskull.woff
# Removing invalid font file: ._SleeplessonPlus-Regular.ttf
# Removing invalid font file: ._Beth-Ellen-2.0.ttf
# Removing invalid font file: ._FB-MommaHero2.otf
# Removing invalid font file: ._Lovely.otf
# Removing invalid font file: ._Halgonak.otf
# Removing non-font file: Font Barnies Free Trial.woff
# Removing invalid font file: ._Beleriand.otf
# Removing invalid font file: ._GelPenUprightLight.ttf
# Removing invalid font file: ._Blackway Brush.otf
# Removing non-font file: black venom.woff
# Removing non-font file: armed-webfont.woff2
# Removing invalid font file: ._Giraffenhals_extended.otf
# Removing invalid font file: ._Walytime.otf
# Removing invalid font file: ._swingdevilDEMO.otf
# Removing invalid font file: ._Bellamy Signature.ttf
# Removing non-font file: Polaria.woff
# Removing non-font file: Beleriand.woff
# Removing invalid font file: ._She Speak.otf
# Removing invalid font file: ._DecalkBold.ttf
# Removing non-font file: Ring singularity.woff
# Removing non-font file: BellisyaSignature.woff
# Removing non-font file: Gomgom Handwrite-Basic.woff
# Removing invalid font file: ._AGATRG__.TTF
# Removing invalid font file: ._Hunthers Dwayne.otf
# Removing invalid font file: ._Gotten.otf
# Removing non-font file: ._froggyprincess-regular.woff
# Removing non-font file: ._fabfeltscript-bold.eot
# Removing invalid font file: ._Chisel Mark.ttf
# Removing invalid font file: ._Rossana-Regular.otf
# Removing invalid font file: ._AgendaKing.ttf
# Removing invalid font file: ._Adora Chalie.ttf
# Removing invalid font file: ._Carolina Hills Personal Use.ttf
# Removing non-font file: coconutz.woff2
# Removing invalid font file: ._FBMightySpiky.otf
# Removing invalid font file: ._PajamaPantsLight.ttf
# Removing invalid font file: ._Amore.otf
# Removing non-font file: Thorletto (free).Otf
# Removing invalid font file: ._GelPenSerifHeavy.ttf
# Removing invalid font file: ._Barlen.ttf
# Removing non-font file: Ragellia Mellinda.svg
# Removing invalid font file: ._SeindahCinttya.ttf
# Removing invalid font file: ._bathi-italic.ttf
# Removing non-font file: Snowy Holiday.woff2
# Removing non-font file: ._Sugarly.woff
# Removing invalid font file: ._Jarida.otf
# Removing non-font file: Megalhaes.woff
# Removing non-font file: Ruth Daisi.woff2
# Removing invalid font file: ._PrinsesstartaBoldItalicDEMO.ttf
# Removing invalid font file: ._DirlyBelly-Regular.ttf
# Removing invalid font file: ._deSolidia.otf
# Removing non-font file: Sketch Wall 1.woff
# Removing invalid font file: ._straphangDEMO.otf
# Removing non-font file: Aliters.woff
# Removing non-font file: southpawwebfont.svg
# Removing invalid font file: ._Tantamount DEMO.otf
# Removing invalid font file: ._Mystical Eyes Personal Use.ttf
# Removing non-font file: Amulman.svg
# Removing non-font file: coconutz.woff
# Removing non-font file: Ruth Daisi.woff
# Removing invalid font file: ._GelPenSerif.ttf
# Removing invalid font file: ._Bimbo-Sharpie-trial.ttf
# Removing invalid font file: ._PrettyRosseSwash.otf
# Removing invalid font file: ._Familior-Swash.otf
# Removing invalid font file: ._Capable of Loving Personal Use.ttf
# Removing invalid font file: ._Marriage Moment Personal Use.ttf
# Removing invalid font file: ._New Balance Personal Use.ttf
# Removing invalid font file: ._KomixCon Italic.otf
# Removing invalid font file: ._Berton-Roman-trial.ttf
# Removing invalid font file: ._DRAGONFLYsaji.ttf
# Removing non-font file: Font Barnies Free Trial.woff2
# Removing non-font file: Darkest Saturday Outline.woff2
# Removing invalid font file: ._StopYelling-Regular.ttf
# Removing invalid font file: ._Bunch of Flowers Personal Use.ttf
# Removing non-font file: Slash Bold.woff
# Removing invalid font file: ._Breakfast on the beach Personal Use.ttf
# Removing invalid font file: ._Grovflab DEMO.otf
# Removing non-font file: SebladeBlackRegular2R.woff
# Removing non-font file: ._fabfeltscript-bold.svg
# Removing invalid font file: ._RomantikaHidup.otf
# Removing invalid font file: ._KomixCon-Bold Italic.ttf
# Removing invalid font file: ._Allison_Script.otf
# Removing non-font file: Tribista.woff2
# Removing invalid font file: ._KLEECS__.ttf
# Removing invalid font file: ._Chapter One Personal Use.ttf
# Removing non-font file: LeagueScriptNumberOne-webfont.svg
# Removing invalid font file: ._NoteToSelf-Regular.ttf
# Removing invalid font file: ._OlivesOutline.otf
# Removing non-font file: fabfeltscript-bold.woff2
# Removing invalid font file: ._Mollitha.ttf
# Removing invalid font file: ._Cloud Control V2.ttf
# Removing invalid font file: ._Beauty Bright Personal Use.ttf
# Removing invalid font file: ._FBMightySpiky.ttf
# Removing invalid font file: ._Freehand-Blockletter-Regular-trial.ttf
# Removing invalid font file: ._Cassandra Personal Use.ttf
# Removing invalid font file: ._Gladyss-Extras.otf
# Removing invalid font file: ._Ronet DEMO.ttf
# Removing invalid font file: ._Raggedways Regular.otf
# Removing non-font file: Sundae Bite Free Trial.woff2
# Removing invalid font file: ._ToastedCinnamon.ttf
# Removing invalid font file: ._Shaky Hand Some Comic.otf
# Removing invalid font file: ._Ratih Hyun.ttf
# Removing invalid font file: ._PrettyRosse.otf
# Removing non-font file: Baby Crab.woff
# Removing invalid font file: ._Jacyking.ttf
# Removing non-font file: SebladeBlackRegular2Italic.woff
# Removing non-font file: quirky_nots.svg
# Removing invalid font file: ._HeySweety.otf
# Removing invalid font file: ._GelPenHeavy.ttf
# Removing invalid font file: ._Rathury.otf
# Removing invalid font file: ._Bimbo-Ballpoint-trial.ttf
# Removing non-font file: Candyful - Free Trial.woff
# Removing invalid font file: ._Daisuky Fancy.otf
