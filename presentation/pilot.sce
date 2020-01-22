write_codes = true;
active_buttons = 2;

begin;

picture {
   default_code = "fixation";
   bitmap {
		filename = "fixation_cross.png";
   };
   x = 0; y = 0;
} fixation;

TEMPLATE "stimulus.tem" {
word file;
"vssov" "symbols_vssov.png";
"opaali" "word_opaali.png";
"zpcqc" "consonants_zpcqc.png";
"o^ss" "symbols_o^ss.png";
"siitos" "word_siitos.png";
"v^od" "symbols_v^od.png";
};

TEMPLATE "question.tem" {
question file;
"v _ _ _" "symbols_v^od_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"duuri" "word_duuri.png";
"ddvsd" "symbols_ddvsd.png";
"lempo" "word_lempo.png";
"nuha" "word_nuha.png";
"laukka" "word_laukka.png";
"jakaja" "word_jakaja.png";
};

TEMPLATE "question.tem" {
question file;
"_ x _ _ _ _" "word_jakaja_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"räme" "word_räme.png";
"kyhmy" "word_kyhmy.png";
"osd^o" "symbols_osd^o.png";
"dsdoo^" "symbols_dsdoo^.png";
"huorin" "word_huorin.png";
"pidot" "word_pidot.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ d _ _" "word_pidot_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"spmb" "consonants_spmb.png";
"jbntmc" "consonants_jbntmc.png";
"uuhi" "word_uuhi.png";
"akti" "word_akti.png";
"vsd^ds" "symbols_vsd^ds.png";
};

TEMPLATE "question.tem" {
question file;
"v _ _ _ _ _" "symbols_vsd^ds_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"tämä" "word_tämä.png";
"qmncn" "consonants_qmncn.png";
"kussa" "word_kussa.png";
"pamppu" "word_pamppu.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ _ _ q _" "word_pamppu_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"tvzdg" "consonants_tvzdg.png";
"czkdbs" "consonants_czkdbs.png";
"rjfcdx" "consonants_rjfcdx.png";
"tyköä" "word_tyköä.png";
"rtsgdh" "consonants_rtsgdh.png";
"vlptrn" "consonants_vlptrn.png";
"nczkbj" "consonants_nczkbj.png";
"vdvv" "symbols_vdvv.png";
"dfdzj" "consonants_dfdzj.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ _ j _" "consonants_dfdzj_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"xpfpj" "consonants_xpfpj.png";
"vzkt" "consonants_vzkt.png";
"silaus" "word_silaus.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ _ d _ _" "word_silaus_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"vdod^" "symbols_vdod^.png";
"lähi" "word_lähi.png";
"scgxmk" "consonants_scgxmk.png";
"pitäen" "word_pitäen.png";
"lotja" "word_lotja.png";
"ovdvd" "symbols_ovdvd.png";
};

TEMPLATE "question.tem" {
question file;
"_ v _ _ _" "symbols_ovdvd_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"kiuas" "word_kiuas.png";
"lltkw" "consonants_lltkw.png";
"säie" "word_säie.png";
"ajos" "word_ajos.png";
"zkwnr" "consonants_zkwnr.png";
"jymy" "word_jymy.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ _ y" "word_jymy_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"almu" "word_almu.png";
"gthb" "consonants_gthb.png";
"zqshvj" "consonants_zqshvj.png";
"jpmxc" "consonants_jpmxc.png";
"wlpn" "consonants_wlpn.png";
"oinas" "word_oinas.png";
"pokeri" "word_pokeri.png";
"whpw" "consonants_whpw.png";
};

TEMPLATE "question.tem" {
question file;
"_ h _ _" "consonants_whpw_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"ldtg" "consonants_ldtg.png";
"ssdvo^" "symbols_ssdvo^.png";
"jxfm" "consonants_jxfm.png";
"ässä" "word_ässä.png";
"näppy" "word_näppy.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ u _ _" "word_näppy_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"fotoni" "word_fotoni.png";
"svvssv" "symbols_svvssv.png";
"bfkx" "consonants_bfkx.png";
"sysi" "word_sysi.png";
"sdsdoo" "symbols_sdsdoo.png";
"uoma" "word_uoma.png";
"psalmi" "word_psalmi.png";
"nrxlzx" "consonants_nrxlzx.png";
};

TEMPLATE "question.tem" {
question file;
"_ r _ _ _ _" "consonants_nrxlzx_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"tcftcg" "consonants_tcftcg.png";
"menijä" "word_menijä.png";
"rukki" "word_rukki.png";
"vvo^" "symbols_vvo^.png";
"nirso" "word_nirso.png";
};

TEMPLATE "question.tem" {
question file;
"_ d _ _ _" "word_nirso_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"^^svd" "symbols_^^svd.png";
"osdoo^" "symbols_osdoo^.png";
"tyrä" "word_tyrä.png";
"fbtsr" "consonants_fbtsr.png";
"vv^d" "symbols_vv^d.png";
};

TEMPLATE "question.tem" {
question file;
"v _ _ _" "symbols_vv^d_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"vvv^" "symbols_vvv^.png";
"blcrm" "consonants_blcrm.png";
"nlmlh" "consonants_nlmlh.png";
"iäti" "word_iäti.png";
"vipu" "word_vipu.png";
"odote" "word_odote.png";
"s^vds^" "symbols_s^vds^.png";
"tcfwdr" "consonants_tcfwdr.png";
};

TEMPLATE "question.tem" {
question file;
"_ c _ _ _ _" "consonants_tcfwdr_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"qgtqmd" "consonants_qgtqmd.png";
"dd^ds^" "symbols_dd^ds^.png";
"hioa" "word_hioa.png";
"säle" "word_säle.png";
"sqdq" "consonants_sqdq.png";
"sv^vvs" "symbols_sv^vvs.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ _ v _ _" "symbols_sv^vvs_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"sdsvs" "symbols_sdsvs.png";
"kopio" "word_kopio.png";
"gkzpg" "consonants_gkzpg.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ _ _ z" "consonants_gkzpg_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"kerubi" "word_kerubi.png";
"katodi" "word_katodi.png";
"oosvsd" "symbols_oosvsd.png";
"mäntä" "word_mäntä.png";
"ztlrlf" "consonants_ztlrlf.png";
"lwzsh" "consonants_lwzsh.png";
"pmxwht" "consonants_pmxwht.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ c _ _ _" "consonants_pmxwht_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"afasia" "word_afasia.png";
"druidi" "word_druidi.png";
"nurja" "word_nurja.png";
"viipyä" "word_viipyä.png";
"ositus" "word_ositus.png";
};

TEMPLATE "question.tem" {
question file;
"ö _ _ _ _ _" "word_ositus_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"dv^oo" "symbols_dv^oo.png";
"dzvxq" "consonants_dzvxq.png";
"känsä" "word_känsä.png";
"kyteä" "word_kyteä.png";
"fuksi" "word_fuksi.png";
"^d^oos" "symbols_^d^oos.png";
"xxfmhj" "consonants_xxfmhj.png";
"sdvd^" "symbols_sdvd^.png";
};

TEMPLATE "question.tem" {
question file;
"_ d _ _ _" "symbols_sdvd^_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"fgrhzh" "consonants_fgrhzh.png";
"eliö" "word_eliö.png";
"puida" "word_puida.png";
"gongi" "word_gongi.png";
"cnwh" "consonants_cnwh.png";
"ylkä" "word_ylkä.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ k _" "word_ylkä_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"itää" "word_itää.png";
"uuni" "word_uuni.png";
"riimu" "word_riimu.png";
"odvs^" "symbols_odvs^.png";
"luusto" "word_luusto.png";
"tuohus" "word_tuohus.png";
"^d^vs" "symbols_^d^vs.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ ^ _ _" "symbols_^d^vs_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"d^oood" "symbols_d^oood.png";
"zdvv" "consonants_zdvv.png";
"grgdd" "consonants_grgdd.png";
"dwmdd" "consonants_dwmdd.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ _ _ d" "consonants_dwmdd_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"diodi" "word_diodi.png";
"ilmi" "word_ilmi.png";
"o^^od" "symbols_o^^od.png";
"rgkpb" "consonants_rgkpb.png";
"nieriä" "word_nieriä.png";
"sikhi" "word_sikhi.png";
};

TEMPLATE "question.tem" {
question file;
"h _ _ _ _" "word_sikhi_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"qpvbs" "consonants_qpvbs.png";
"kolvi" "word_kolvi.png";
"dfszzp" "consonants_dfszzp.png";
"sdosd" "symbols_sdosd.png";
"kihu" "word_kihu.png";
"emätin" "word_emätin.png";
};

TEMPLATE "question.tem" {
question file;
"c _ _ _ _ _" "word_emätin_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"sovo" "symbols_sovo.png";
"ctzdzl" "consonants_ctzdzl.png";
"vaje" "word_vaje.png";
"mgttwx" "consonants_mgttwx.png";
"dovdd" "symbols_dovdd.png";
"^odds" "symbols_^odds.png";
"solmio" "word_solmio.png";
"suippo" "word_suippo.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ _ p _ _" "word_suippo_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"sdvsv" "symbols_sdvsv.png";
"vsds" "symbols_vsds.png";
"piip" "word_piip.png";
"motata" "word_motata.png";
"ovdod^" "symbols_ovdod^.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ _ _ d _" "symbols_ovdod^_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"qvkjt" "consonants_qvkjt.png";
"vovso" "symbols_vovso.png";
"pzsfrc" "consonants_pzsfrc.png";
"ilkiö" "word_ilkiö.png";
"^s^d" "symbols_^s^d.png";
"osvs^d" "symbols_osvs^d.png";
"xnbrh" "consonants_xnbrh.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ _ r _" "consonants_xnbrh_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"ähky" "word_ähky.png";
"lakea" "word_lakea.png";
"^dod^" "symbols_^dod^.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ _ _ s" "symbols_^dod^_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"kolhia" "word_kolhia.png";
"jdfjs" "consonants_jdfjs.png";
"yöpyä" "word_yöpyä.png";
"jäte" "word_jäte.png";
"wqghh" "consonants_wqghh.png";
"kuje" "word_kuje.png";
"dod^os" "symbols_dod^os.png";
};

TEMPLATE "question.tem" {
question file;
"_ v _ _ _ _" "symbols_dod^os_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"häpy" "word_häpy.png";
"vd^s" "symbols_vd^s.png";
"räntä" "word_räntä.png";
"klaava" "word_klaava.png";
"qjtnl" "consonants_qjtnl.png";
"ampuja" "word_ampuja.png";
"vo^^" "symbols_vo^^.png";
"kaapia" "word_kaapia.png";
};

TEMPLATE "question.tem" {
question file;
"k _ _ _ _ _" "word_kaapia_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"otsoni" "word_otsoni.png";
"v^dv^s" "symbols_v^dv^s.png";
"s^vo" "symbols_s^vo.png";
};

TEMPLATE "question.tem" {
question file;
"_ ^ _ _" "symbols_s^vo_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"ahjo" "word_ahjo.png";
"osvv" "symbols_osvv.png";
"miten" "word_miten.png";
"zsgj" "consonants_zsgj.png";
"faksi" "word_faksi.png";
"qgjwzq" "consonants_qgjwzq.png";
"erkani" "word_erkani.png";
"lohi" "word_lohi.png";
"syöpyä" "word_syöpyä.png";
};

TEMPLATE "question.tem" {
question file;
"_ a _ _ _ _" "word_syöpyä_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"möly" "word_möly.png";
"^dso" "symbols_^dso.png";
"uute" "word_uute.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ t _" "word_uute_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"salvaa" "word_salvaa.png";
"ssvv" "symbols_ssvv.png";
"zfjxqk" "consonants_zfjxqk.png";
"gcfj" "consonants_gcfj.png";
"salaus" "word_salaus.png";
"tenä" "word_tenä.png";
"bwdz" "consonants_bwdz.png";
"poru" "word_poru.png";
"kustos" "word_kustos.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ ü _ _ _" "word_kustos_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"vvo^ss" "symbols_vvo^ss.png";
"loraus" "word_loraus.png";
"syaani" "word_syaani.png";
};

TEMPLATE "question.tem" {
question file;
"s _ _ _ _ _" "word_syaani_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"nurin" "word_nurin.png";
"isyys" "word_isyys.png";
"harjus" "word_harjus.png";
"osinko" "word_osinko.png";
"kysta" "word_kysta.png";
"pesula" "word_pesula.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ l _ _ _" "word_pesula_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"älytä" "word_älytä.png";
"^so^d" "symbols_^so^d.png";
"silsa" "word_silsa.png";
"vs^vds" "symbols_vs^vds.png";
"dovv^" "symbols_dovv^.png";
"nvvvz" "consonants_nvvvz.png";
"xxqhnz" "consonants_xxqhnz.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ _ _ n _" "consonants_xxqhnz_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"tykö" "word_tykö.png";
"äimä" "word_äimä.png";
"wvxw" "consonants_wvxw.png";
"voov^v" "symbols_voov^v.png";
"dvsvos" "symbols_dvsvos.png";
"urut" "word_urut.png";
};

TEMPLATE "question.tem" {
question file;
"u _ _ _" "word_urut_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"svds" "symbols_svds.png";
"näkö" "word_näkö.png";
"s^dv^" "symbols_s^dv^.png";
"^sodvv" "symbols_^sodvv.png";
"ripsi" "word_ripsi.png";
"myyty" "word_myyty.png";
"tyvi" "word_tyvi.png";
"^o^o^" "symbols_^o^o^.png";
};

TEMPLATE "question.tem" {
question file;
"_ v _ _ _" "symbols_^o^o^_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"fzphjd" "consonants_fzphjd.png";
"sclc" "consonants_sclc.png";
"rulla" "word_rulla.png";
"uima" "word_uima.png";
"^s^s" "symbols_^s^s.png";
"äänes" "word_äänes.png";
};

TEMPLATE "question.tem" {
question file;
"_ ä _ _ _" "word_äänes_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"noppa" "word_noppa.png";
"gnxd" "consonants_gnxd.png";
"korren" "word_korren.png";
"estyä" "word_estyä.png";
"ssss" "symbols_ssss.png";
"uivelo" "word_uivelo.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ _ _ _ o" "word_uivelo_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"mplr" "consonants_mplr.png";
"d^sd" "symbols_d^sd.png";
"orpo" "word_orpo.png";
"hamaan" "word_hamaan.png";
"juhta" "word_juhta.png";
"ttjglk" "consonants_ttjglk.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ _ _ _ r" "consonants_ttjglk_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"hlsp" "consonants_hlsp.png";
"doosv" "symbols_doosv.png";
"köli" "word_köli.png";
"sodvds" "symbols_sodvds.png";
"terska" "word_terska.png";
"do^so" "symbols_do^so.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ d _ _" "symbols_do^so_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"kphwjz" "consonants_kphwjz.png";
"^osd" "symbols_^osd.png";
"vuoka" "word_vuoka.png";
"kortti" "word_kortti.png";
"mxnhh" "consonants_mxnhh.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ _ _ h" "consonants_mxnhh_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"päkiä" "word_päkiä.png";
"kytkin" "word_kytkin.png";
"qckq" "consonants_qckq.png";
"lldk" "consonants_lldk.png";
"d^^do" "symbols_d^^do.png";
"btmk" "consonants_btmk.png";
"sävy" "word_sävy.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ e _" "word_sävy_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"hcpjtf" "consonants_hcpjtf.png";
"tlzr" "consonants_tlzr.png";
"sodo" "symbols_sodo.png";
"hihna" "word_hihna.png";
"rämä" "word_rämä.png";
};

TEMPLATE "question.tem" {
question file;
"_ h _ _" "word_rämä_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"kopina" "word_kopina.png";
"^dvd" "symbols_^dvd.png";
"sovdvo" "symbols_sovdvo.png";
"vdso" "symbols_vdso.png";
"torium" "word_torium.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ _ _ _ m" "word_torium_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"wnnz" "consonants_wnnz.png";
"pmrh" "consonants_pmrh.png";
"crtw" "consonants_crtw.png";
"pyörö" "word_pyörö.png";
"uiva" "word_uiva.png";
};

TEMPLATE "question.tem" {
question file;
"d _ _ _" "word_uiva_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"sarake" "word_sarake.png";
"^s^sso" "symbols_^s^sso.png";
"hajan" "word_hajan.png";
"kohu" "word_kohu.png";
"lemu" "word_lemu.png";
"hqzll" "consonants_hqzll.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ _ l _" "consonants_hqzll_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"arpoa" "word_arpoa.png";
"pöty" "word_pöty.png";
"häkä" "word_häkä.png";
"sdvdvd" "symbols_sdvdvd.png";
"uuma" "word_uuma.png";
"^o^so" "symbols_^o^so.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ _ s _" "symbols_^o^so_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"zztr" "consonants_zztr.png";
"särkyä" "word_särkyä.png";
"häät" "word_häät.png";
"jtltfj" "consonants_jtltfj.png";
"jaos" "word_jaos.png";
"ääriin" "word_ääriin.png";
"dso^" "symbols_dso^.png";
};

TEMPLATE "question.tem" {
question file;
"d _ _ _" "symbols_dso^_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"osuma" "word_osuma.png";
"txhs" "consonants_txhs.png";
"itiö" "word_itiö.png";
"zjwhs" "consonants_zjwhs.png";
"gsdx" "consonants_gsdx.png";
"bkcvxf" "consonants_bkcvxf.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ _ b _ _" "consonants_bkcvxf_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"säiky" "word_säiky.png";
"ovods" "symbols_ovods.png";
"ratamo" "word_ratamo.png";
"hormi" "word_hormi.png";
"sopu" "word_sopu.png";
};

TEMPLATE "question.tem" {
question file;
"_ c _ _" "word_sopu_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"ripeä" "word_ripeä.png";
"vsd^" "symbols_vsd^.png";
"^sdsov" "symbols_^sdsov.png";
"jaardi" "word_jaardi.png";
"vahaus" "word_vahaus.png";
"kuskus" "word_kuskus.png";
"pesin" "word_pesin.png";
"^svdvv" "symbols_^svdvv.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ _ _ _ d" "symbols_^svdvv_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"dv^od" "symbols_dv^od.png";
"o^v^" "symbols_o^v^.png";
"rapsi" "word_rapsi.png";
"rrggj" "consonants_rrggj.png";
"hautua" "word_hautua.png";
"oo^^d^" "symbols_oo^^d^.png";
};

TEMPLATE "question.tem" {
question file;
"^ _ _ _ _ _" "symbols_oo^^d^_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"vssd" "symbols_vssd.png";
"s^s^s" "symbols_s^s^s.png";
"stoola" "word_stoola.png";
"sddo" "symbols_sddo.png";
"seimi" "word_seimi.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ _ m _" "word_seimi_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"^s^^" "symbols_^s^^.png";
"karies" "word_karies.png";
"jqdsmj" "consonants_jqdsmj.png";
"äänne" "word_äänne.png";
"suti" "word_suti.png";
"dodv" "symbols_dodv.png";
"cmdhv" "consonants_cmdhv.png";
};

TEMPLATE "question.tem" {
question file;
"c _ _ _ _" "consonants_cmdhv_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"^vd^vd" "symbols_^vd^vd.png";
"shiia" "word_shiia.png";
"nqxzt" "consonants_nqxzt.png";
"kaste" "word_kaste.png";
"säkä" "word_säkä.png";
"jänne" "word_jänne.png";
"nisä" "word_nisä.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ _ y" "word_nisä_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
"someen" "word_someen.png";
"täällä" "word_täällä.png";
"dssddd" "symbols_dssddd.png";
"nide" "word_nide.png";
"^ooso" "symbols_^ooso.png";
"gljt" "consonants_gljt.png";
};

TEMPLATE "question.tem" {
question file;
"_ _ _ t" "consonants_gljt_question.png";
};

TEMPLATE "stimulus.tem" {
word file;
};
